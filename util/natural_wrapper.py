import numpy as np
import gymnasium as gym
import cv2
import glob
import random
import os

# --- Image Sources ---

class ImageSource:
    """Base class for background sources."""
    def get_image(self):
        """Returns an RGB image of [h, w, 3]."""
        raise NotImplementedError()

    def reset(self):
        """Called when an episode ends."""
        pass

class NoiseSource(ImageSource):
    """Generates random Gaussian noise backgrounds."""
    def __init__(self, shape, strength=255):
        self.shape = shape
        self.strength = strength

    def get_image(self):
        # Generate noise in range [0, 1] then scale to strength
        noise = np.random.rand(self.shape[0], self.shape[1], 3) * self.strength
        return noise.astype(np.uint8)

class VideoSource(ImageSource):
    """
    Plays a video as the background using OpenCV.
    """
    def __init__(self, shape, file_path, videos_folder=None):
        self.shape = shape
        self.videos_folder = videos_folder
        self.current_file_path = file_path
        self._load_video(file_path)

    def _load_video(self, file_path):
        """Load a video file."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(file_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {file_path}")
        
        # Get video info
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

    def _get_random_video(self):
        """Get a random video file from the videos folder."""
        if self.videos_folder is None:
            return self.current_file_path
            
        video_pattern = os.path.join(self.videos_folder, '**', '*.mp4')
        video_files = glob.glob(video_pattern, recursive=True)
        
        if not video_files:
            return self.current_file_path
        
        # Try to get a different video than the current one
        available_videos = [v for v in video_files if v != self.current_file_path]
        if not available_videos:
            available_videos = video_files
            
        selected_video = random.choice(available_videos)
        return selected_video

    def get_image(self):
        ret, frame = self.cap.read()
        
        if not ret:
            # Video ended, load a new random video
            if self.videos_folder is not None:
                new_video = self._get_random_video()
                self.current_file_path = new_video
                self._load_video(new_video)
            else:
                # No videos folder, restart current video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
            ret, frame = self.cap.read()
            
            if not ret:
                # If still can't read, return black frame
                return np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint8)
        
        # Resize video frame to match environment observation size if needed
        if frame.shape[:2] != self.shape:
            frame = cv2.resize(frame, (self.shape[1], self.shape[0]))
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        return frame

    def reset(self):
        # Optional: Randomize start position in video on reset?
        # For now, just restart from beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# --- Matting Strategies ---

class BackgroundMatting:
    def get_mask(self, img):
        raise NotImplementedError()

class BackgroundMattingWithColor(BackgroundMatting):
    """
    Identifies background by a specific color (e.g., Black for Breakout).
    Returns a mask where 1 = background, 0 = foreground.
    """
    def __init__(self, color, threshold=0):
        self._color = np.array(color)
        self._threshold = threshold

    def get_mask(self, img):
        # Calculate distance from the target background color
        # img is (H, W, 3), color is (3,)
        # We want to find pixels close to color
        diff = np.abs(img - self._color).sum(axis=2)
        return diff <= self._threshold

# --- The Wrapper ---

class ReplaceBackgroundEnv(gym.ObservationWrapper):
    def __init__(self, env, bg_matting, natural_source):
        super().__init__(env)
        self._bg_matting = bg_matting
        self._natural_source = natural_source
        
    def _replace_background(self, img):
        # 1. Get the mask (True where background is)
        is_background = self._bg_matting.get_mask(img)
        
        # 2. Get the new background image
        bg_image = self._natural_source.get_image()
        
        # 3. Ensure bg_image is the same type/range as observation (uint8)
        if bg_image.dtype != np.uint8:
            bg_image = bg_image.astype(np.uint8)
            
        # 4. Replace pixels
        # Create a copy to avoid modifying the original buffer if it's shared
        new_img = img.copy()
        
        # Ensure shapes match (in case source returns slightly different size)
        if bg_image.shape != new_img.shape:
             bg_image = cv2.resize(bg_image, (new_img.shape[1], new_img.shape[0]))

        new_img[is_background] = bg_image[is_background]
        return new_img

    def observation(self, observation):
        return self._replace_background(observation)

    def render(self):
        frame = self.env.render()
        if frame is None:
            return None
        # Only apply replacement if the frame looks like an observation (RGB)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
             return self._replace_background(frame)
        return frame

    def reset(self, **kwargs):
        self._natural_source.reset()
        return super().reset(**kwargs)
