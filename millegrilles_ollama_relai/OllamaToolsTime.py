import logging
import pytz

from datetime import datetime, timezone
from typing import Optional

from millegrilles_ollama_relai.ToolStructs import OllamaTool

LOGGER = logging.getLogger(__name__)

class ToolTime(OllamaTool):

    def __init__(self):
        super().__init__()
        self.__setup()

    def __setup(self):
        self._tools.append({
            'type': 'function',
            'function': {
                'name': f'{self.name}_get_current_utc_date_and_time',
                'description': 'Returns the current UTC date and time',
            },
            'call': _get_current_utc_date_and_time,
        })
        self._tools.append({
            'type': 'function',
            'function': {
                'name': f'{self.name}_get_current_user_date_and_time',
                'description': 'Returns the current user\'s date and time with appropriate timezone',
                'parameters': {
                    'type': 'object',
                    'required': [],
                    'properties': {
                        'tz': {'type': 'str', 'description': 'pytz timezone. Optional, used to override the user profile.'},
                    },
                },
            },
            'call': _get_current_user_date_and_time,
        })

    @property
    def name(self) -> str:
        return "time"


def _get_current_utc_date_and_time(*args, **kwargs) -> str:
    now = datetime.now(tz=pytz.UTC).strftime("%a, %d %b %Y %H:%M:%S")
    LOGGER.debug(f"_get_current_utc_date_and_time: {now} UTC")
    return f"The current date and time is: {now} UTC."

def _get_current_user_date_and_time(user_profile: dict, tz: Optional[str] = None, *args, **kwargs) -> str:
    user_timezone_str = tz or user_profile.get('timezone') or 'America/Montreal'
    user_timezone = pytz.timezone(user_timezone_str)
    now = datetime.now(tz=user_timezone)
    LOGGER.debug(f"_get_current_user_date_and_time: {now.strftime("%a, %d %b %Y %H:%M:%S %Z")}")
    return f"The user's current date and time is: {now.strftime("%a, %d %b %Y %H:%M:%S %Z")}"
