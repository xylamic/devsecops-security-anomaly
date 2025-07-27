from enum import IntEnum
from datetime import datetime
import sys
import json

class AnomalousUser:

    class AnomalyType(IntEnum):
        ZSCORE = 1
        NINETYFIVE = 2
        NINETYNINE = 3
        FIXED = 4

    actor: str
    day: datetime
    anomaly_score: float
    email: str
    anomaly_features: list[tuple[str, any, AnomalyType, any]]

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], str):
            # Constructor for JSON string
            self.__dict__ = json.loads(args[0])
            if 'anomalies' not in self.__dict__:
                self.anomaly_features = []
        elif len(args) == 4 and isinstance(args[0], str) and isinstance(args[1], datetime) and isinstance(args[2], float):
            # Constructor for individual attributes
            self.actor = args[0]
            self.day = args[1]
            self.anomaly_score = args[2]
            self.email = str(args[3])
            self.anomaly_features = []
        else:
            raise ValueError("Invalid arguments for AnomalousUser constructor")

    
    def add_anomaly(self, column: str, value: any, anomaly_type: AnomalyType, threshold: any):
        self.anomaly_features.append((column, value, anomaly_type, threshold))

    def __str__(self):
        return f"Actor: {self.actor}, Day: {self.day}, Anomaly Score: {self.anomaly_score}, Email: {self.email}"
    

    @staticmethod
    def read_anomalies_from_file(file: str):
        # read in the json file as objects
         with open(file, 'r') as f:
            users = [AnomalousUser(json.dumps(user)) for user in json.load(f)]
            return users
         

    @staticmethod
    def write_anomalies_to_file(ausers: 'list[AnomalousUser]', file: str):
        ausers_json = json.dumps([auser.__dict__ for auser in ausers], default=str, indent=4)

        with open(file, 'w') as file:
            file.write(ausers_json)


if __name__ in {"__main__", "__mp_main__"}:

    input_file = "../data/anomalies.json"

    # read in the json file as objects
    users = AnomalousUser.read_anomalies_from_file(input_file)

    # group by actor
    actors = {}
    for user in users:
        if user.actor not in actors:
            actors[user.actor] = []
        actors[user.actor].append(user)
    
    # sort each actor's anomalies by day
    for actor in actors:
        actors[actor].sort(key=lambda x: x.day)
    
    from nicegui import ui

    css = """
        <style>
        .table-condensed td, .table-condensed th {
            padding: 2px; /* Adjust the padding as needed */
            font-size: 12px; /* Optional: Adjust the font size */
        }
        </style>
        """
    
    # Inject custom CSS
    ui.html(css)

    # for each actor, create a table of anomalies for each day
    for actor in actors:
        ui.label(f"Actor: {actor}")

        for user in actors[actor]:
            ui.label(f"Day: {user.day}")
            columns = [
                {'name': 'column', 'label': 'Column', 'field': 'column', 'sortable': True},
                {'name': 'value', 'label': 'Value', 'field': 'value', 'sortable': True},
                {'name': 'anomaly_type', 'label': 'Anomaly Type', 'field': 'anomaly_type', 'sortable': True},
                {'name': 'threshold', 'label': 'Threshold', 'field': 'threshold', 'sortable': True}
            ]
            rows = [{'column': anomaly[0], 'value': anomaly[1], 'anomaly_type': anomaly[2], 'threshold': anomaly[3]} for anomaly in user.anomalies]
            ui.table(columns=columns, rows=rows, row_key='column').classes('table-condensed')
        
        ui.html('<hr />')

    ui.run()