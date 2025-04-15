import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class for the application."""
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(24))
    DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')
    
    # Application paths
    APP_DIR = os.path.abspath(os.path.dirname(__file__))
    RESULTS_DIR = os.path.join(APP_DIR, 'results')
    
    # API configurations
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o')
    
    # Task settings
    TASK_TIMEOUT = int(os.environ.get('TIMEOUT', 300))
    
    @classmethod
    def init_app(cls, app):
        """Initialize the Flask application with this configuration."""
        app.config.from_object(cls)
        
        # Ensure the results directory exists
        if not os.path.exists(cls.RESULTS_DIR):
            os.makedirs(cls.RESULTS_DIR)


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    WTF_CSRF_ENABLED = False
    RESULTS_DIR = os.path.join(Config.APP_DIR, 'test_results')


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
