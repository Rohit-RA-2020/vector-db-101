import os
import re
import json
import whisper
import subprocess
from pytubefix import Playlist
from fastapi import FastAPI, HTTPException

app = FastAPI()

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def download_playlist(playlist_url):
    playlist = Playlist(playlist_url)
    print(f'Downloading playlist: {playlist.title}')
    
    video_data = []
    for video in playlist.videos:
        sanitized_title = sanitize_filename(video.title)
        print(f'Downloading video: {sanitized_title}')
        video.streams.filter(only_audio=True).first().download(filename=f"{sanitized_title}.mp4")
        video_data.append({
            'filename': f"{sanitized_title}.mp4",
            'video_id': video.video_id,
            'playlist_url': playlist_url
        })
    
    return video_data

def extract_audio(video_data):
    audio_files = []
    for video in video_data:
        video_file = video['filename']
        print(f'Extracting audio from {video_file}')
        audio_file = video_file.replace('.mp4', '.mp3')
        
        try:
            command = [
                'ffmpeg',
                '-i', video_file,
                '-vn',
                '-acodec', 'libmp3lame',
                '-ab', '192k',
                '-ar', '44100',
                '-y',
                audio_file
            ]
            
            subprocess.run(command, check=True, capture_output=True)
            video['audio_file'] = audio_file
            audio_files.append(video)
            
            os.remove(video_file)
            
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio from {video_file}: {e.stderr.decode()}")
            continue
        except Exception as e:
            print(f"Unexpected error processing {video_file}: {str(e)}")
            continue
    
    return audio_files

def construct_video_url(video_id, playlist_url, timestamp):
    # Convert timestamp to seconds (round down)
    seconds = int(timestamp)
    return f"https://www.youtube.com/watch?v={video_id}&list={playlist_url.split('list=')[1]}&t={seconds}"

def transcribe_audio(audio_files):
    model = whisper.load_model("base")
    transcripts = {}
    
    for video in audio_files:
        audio_file = video['audio_file']
        print(f'Transcribing {audio_file}')
        result = model.transcribe(audio_file, word_timestamps=True)
        
        transcripts[audio_file] = []
        for segment in result['segments']:
            transcripts[audio_file].append({
                'text': segment['text'],
                'start': segment['start'],
                'end': segment['end'],
                'url': construct_video_url(
                    video_id=video['video_id'],
                    playlist_url=video['playlist_url'],
                    timestamp=segment['start']
                )
            })
        
        transcript_file = audio_file.replace('.mp3', '_transcript.json')
        with open(transcript_file, 'w') as f:
            json.dump(transcripts[audio_file], f, indent=4)
            
        print(f'Transcription saved to {transcript_file}')
        
        os.remove(audio_file)
        
    return transcripts

@app.post("/transcribe-playlist/")
async def transcribe_playlist(playlist_url: str):
    if not playlist_url:
        raise HTTPException(status_code=400, detail="Playlist URL is required.")
    
    if not os.path.exists('transcripts'):
        os.makedirs('transcripts')
    
    os.chdir('transcripts')
    
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise HTTPException(status_code=500, detail="FFmpeg is not installed on the system.")
    
    video_data = download_playlist(playlist_url)
    audio_files = extract_audio(video_data)
    transcripts = transcribe_audio(audio_files)

    return {"transcripts": transcripts}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)