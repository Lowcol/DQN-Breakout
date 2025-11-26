import numpy as np
import gymnasium as gym
import cv2

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
    Plays a video as the background. 
    Requires: pip install scikit-video
    """
    def __init__(self, shape, file_path):
        try:
            import skvideo.io
        except ImportError:
            raise ImportError("Please install scikit-video to use VideoSource: pip install scikit-video")
            
        self.shape = shape
        self.data = skvideo.io.vread(file_path)
        self.i = 0
        print(f"Loaded video {file_path} with shape {self.data.shape}")

    def get_image(self):
        if self.i >= len(self.data):
            self.i = 0
        img = self.data[self.i]
        self.i += 1
        
        # Resize video frame to match environment observation size if needed
        if img.shape[:2] != self.shape:
            img = cv2.resize(img, (self.shape[1], self.shape[0]))
            
        return img

    def reset(self):
        # Optional: Randomize start position in video on reset?
        # For now, just keep playing loop
        pass

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
