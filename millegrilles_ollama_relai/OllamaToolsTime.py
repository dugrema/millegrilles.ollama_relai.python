import pytz

from datetime import datetime, timezone

from millegrilles_ollama_relai.ToolStructs import OllamaTool


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
            },
            'call': _get_current_user_date_and_time,
        })

    @property
    def name(self) -> str:
        return "time"


def _get_current_utc_date_and_time() -> str:
    now = datetime.now(tz=pytz.UTC).strftime("%a, %d %b %Y %H:%M:%S")
    # print(f"_get_current_utc_date_and_time: {now} UTC")
    return f"The current date and time is: {now} UTC."

def _get_current_user_date_and_time() -> str:
    now = datetime.now()
    local_timezone = datetime.now(timezone.utc).astimezone().tzinfo
    # print(f"_get_current_user_date_and_time: {now.strftime("%a, %d %b %Y %H:%M:%S")} {local_timezone}")
    return f"The user's current date and time is: {datetime.now().strftime("%a, %d %b %Y %H:%M:%S")} {local_timezone}"
