"""SQLite interaction history, participant isolation, corrections and neural state."""
import json
import sqlite3
import time
import uuid
import numpy as np


class ContactMemory:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path))
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS sessions (
          id TEXT PRIMARY KEY, participant TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events (
          id TEXT PRIMARY KEY, session TEXT NOT NULL REFERENCES sessions(id),
          timestamp REAL NOT NULL, payload TEXT NOT NULL, state TEXT NOT NULL,
          model_version TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS events_session ON events(session,timestamp);
        CREATE TABLE IF NOT EXISTS feedback (
          id INTEGER PRIMARY KEY, event TEXT NOT NULL REFERENCES events(id),
          head TEXT NOT NULL, label TEXT NOT NULL, timestamp REAL NOT NULL);
        ''')

    def create_session(self, participant, session=None):
        if not isinstance(participant,str) or not participant.strip():
            raise ValueError('Participant identifier required')
        session = session or str(uuid.uuid4())
        with self.db:
            self.db.execute('INSERT INTO sessions VALUES (?,?)',(session,participant))
        return session

    def check_session(self, session, participant):
        row=self.db.execute('SELECT participant FROM sessions WHERE id=?',(session,)).fetchone()
        if not row or row[0] != participant:
            raise ValueError('Session does not belong to participant')

    def append(self, session, participant, event, state, model_version):
        self.check_session(session,participant)
        event_id=str(uuid.uuid4())
        with self.db:
            self.db.execute('INSERT INTO events VALUES (?,?,?,?,?,?)',
                (event_id,session,time.time(),json.dumps(event,allow_nan=False),
                 json.dumps(np.asarray(state).tolist(),allow_nan=False),model_version))
        return event_id

    def history(self, session, participant):
        self.check_session(session,participant)
        return [dict(id=r[0],event=json.loads(r[1]),state=np.array(json.loads(r[2])),version=r[3])
                for r in self.db.execute('SELECT id,payload,state,model_version FROM events WHERE session=? ORDER BY rowid',(session,))]

    def correct(self, session, participant, event_id, head, label):
        self.check_session(session,participant)
        if not self.db.execute('SELECT 1 FROM events WHERE id=? AND session=?',(event_id,session)).fetchone():
            raise ValueError('Event not found in this session')
        with self.db:
            self.db.execute('INSERT INTO feedback(event,head,label,timestamp) VALUES (?,?,?,?)',
                            (event_id,head,label,time.time()))

    def training_records(self, session, participant):
        history=self.history(session,participant)
        records=[]
        for index,row in enumerate(history):
            corrections=self.db.execute('SELECT head,label FROM feedback WHERE event=? ORDER BY id',(row['id'],)).fetchall()
            if corrections:
                records.append({'events':[r['event'] for r in history[:index+1]],'targets':dict(corrections)})
        return records

    def close(self):
        self.db.close()
