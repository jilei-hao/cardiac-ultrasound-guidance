from dataclasses import dataclass


@dataclass
class GuidanceUIConfig:
    env_config: dict
    input_source: str
    force_2d_guidance_widget: bool



class GuidanceUI:
    def __init__(self, config):
        self._config = config
        self._env_config = config.env_config
        # UI components
        self._main_window = None
        self._landmark_renderer = None
        self._guidance_widget_renderer = None
        self._application_settings = None
        # Data Access
        self._data_streamer = None
        # Inference Pipeline
        self._guidance_pipeline = None
    
    def initialize(self):
        """Initialize the UI components and inference pipeline."""
        pass


    def run(self):
        """Run the main UI loop."""
        pass


    def shutdown(self):
        """Shutdown the UI and release resources."""
        pass




    

