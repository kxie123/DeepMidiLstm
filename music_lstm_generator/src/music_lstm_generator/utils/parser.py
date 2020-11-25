
import logging
import math
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, interval, pitch, stream

logging.basicConfig(level = logging.INFO)
log = logging.getLogger()

KEYS = ['A','A#','B','C','C#','D','D#','E','F','F#','G','G#']

class MidiFileParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self._time_signatures = None
        self._base_key = None
        try:
            self.midi = converter.parse(file_path)
            #flatten score to single voice for simplicity
            self.midi_part = self.midi.flat
            self._ts_elements = self.midi_part.getElementsByClass('TimeSignature')
        except Exception as e:
            log.exception(e)
    
    @property
    def time_signatures(self, ts):
        if not self._time_signatures:
            if len(ts) == 0:
                raise Exception("No time signature exists!")
            for sig in ts:
                #restrict to simple time signatures
                if sig.numerator!=2 and sig.numerator!=4:
                    raise ValueError("Time signature must be in 2 or 4!")
            self._time_signatures = ts
        return self._time_signatures

    @property
    def base_key(self):
        if not self._base_key:
            self._base_key = self.midi.analyze('key').tonic
        return self._base_key

    def transpose_key(self, key):
        return self.midi_part.transpose(interval.Interval(self.base_key, pitch.Pitch(key)))
    
    def get_normalized_notes(self, notes_stream, approach):
        if approach == "intval":
            return self._interval_approach(notes_stream)
        elif approach == "stack":
            return self._stack_approach(notes_stream)
        raise ValueError("Approach is invalid!")

    def load_notes(self, notes_path):
        with open(notes_path, 'rb') as filepath:
            notes = pickle.load(filepath) 
        return notes

    def create_pred_midi(self, prediction_output, output_path):
        output_notes = []
        prev_length = 0
        prev_offset=0
        prev_isNote = False
        cur_note = None
        intvals = []
        rest_dict = {'r1':.25,'r2':.5,'r4':1}
        for pattern in prediction_output:
            if pattern.strip('-').isdigit():
                pattern = int(pattern)
                if pattern>0:
                    if cur_note is not None:
                        output_notes,prev_length,prev_offset = self._append_chord_pred(cur_note, intvals, output_notes,prev_length,prev_offset)

                        intvals=[]     
                    cur_note = note.Note(pattern)
                    prev_isNote = True
                else:
                    if prev_isNote:
                        intvals.append(pattern)
            else: #pattern is a rest
                if prev_isNote:
                    
                    chord_notes = []
                    for i in intvals:
                        chord_notes.append(cur_note.transpose(i))

                    if len(chord_notes)==0:
                        new_note = cur_note
                    else:
                        new_note = chord.Chord([cur_note]+chord_notes)

                    new_note.offset = prev_offset+prev_length
                    new_note.quarterLength = 1
                    prev_length = 0.25
                    prev_offset = new_note.offset
                    output_notes.append(new_note)
                    cur_note = None
                    intvals=[]
                    
                new_rest = note.Rest()
                new_rest.offset = prev_offset+prev_length
                new_rest.quarterLength = rest_dict[pattern]
                prev_length = new_rest.quarterLength
                prev_offset = new_rest.offset
                output_notes.append(new_rest)
                prev_isNote = False

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp=output_path)

    def _append_chord_stack(self, note_list, output_notes, prev_length, prev_offset):
        new_note = chord.Chord(note_list)
        if len(note_list)==1:
            new_note = note.Note(note_list[0])
        new_note.offset = prev_offset+prev_length
        new_note.quarterLength = 1

        prev_length = 0.25
        prev_offset = new_note.offset
        
        output_notes.append(new_note)
        note_list=[]
        return note_list,prev_length, prev_offset

    def stream_raw_notes(self, transposition):
        cur_notes = transposition.getElementsByClass(['Note','Chord'])
        #generator into list comprenhension
        cur_notes = [e for e in cur_notes]
        cur_notes = stream.Stream(cur_notes)
        log.debug("Raw notes parsed: {}".format(cur_notes))
        return cur_notes
    
    def _get_max_offset(self, notes_stream):
        return math.ceil(notes_stream.highestOffset)

    def _create_chord_intervals(self, cur_chord):
        chord_intervals = []
        chord_notes = sorted(chord.Chord(cur_chord))
        top_note = chord_notes[-1]

        if len(cur_chord) > 1:
            for p in reversed(chord_notes[:-1]):
                intval = interval.notesToChromatic(top_note.pitch,p.pitch).semitones
                chord_intervals.append(str(intval))

        return chord_intervals, str(top_note.pitch.midi)

    def _append_chord_pred(self, cur_note, intvals, output_notes, prev_length, prev_offset):
        chord_notes = []
        for i in intvals:
            chord_notes.append(cur_note.transpose(i))

        if len(chord_notes)==0:
            new_note = cur_note
        else:
            new_note = chord.Chord([cur_note]+chord_notes)

        new_note.offset = prev_offset+prev_length
        new_note.quarterLength = 1

        prev_length = 0.25
        prev_offset = new_note.offset
        
        output_notes.append(new_note)
        return output_notes, prev_length, prev_offset

    def _interval_approach(self, cur_notes):
        parsed = 0
        normalized_notes = []
        max_off = math.ceil(cur_notes.highestOffset)
        rest_counter = 0
        #rests = {1:'r1',2:'r2'}
        for i in np.arange(0,max_off+0.25,.25):
            step_notes = [e for e in cur_notes.getElementsByOffset(i,i+.25,includeEndBoundary=False)]
            if len(step_notes)==0:
                #append rest if 0 note length
                normalized_notes.append('r1')
                rest_counter = rest_counter+1
                #if 4 rests in a row, represent as 'long rest'
                if (rest_counter==4):
                    normalized_notes.append('r4')
                    rest_counter = 0
            
            elif len(step_notes)==1 and isinstance(step_notes[0], note.Note):
                if rest_counter != 0:
                    normalized_notes.extend(['r1']*rest_counter)
                    rest_counter = 0
                #append pitch of note
                normalized_notes.append(str(step_notes[0].pitch.midi))
            else:
                if rest_counter!= 0:
                    normalized_notes.extend(['r1']*rest_counter)
                    rest_counter = 0
                    
                cur_chord = []
                for i in np.arange(0,len(step_notes)-1):
                    cur_chord.append(step_notes[i])
                    if step_notes[i].offset != step_notes[i+1].offset:
                        chord_construct, top_note = self._create_chord_intervals(cur_chord)
                        normalized_notes.append(top_note)
                        normalized_notes.append(chord_construct)
                        cur_chord=[]   
                #append last element of timestep
                chord_construct, top_note = self._create_chord_intervals(cur_chord+[step_notes[-1]])
                normalized_notes.append(top_note)
                normalized_notes.append(chord_construct)

        parsed = parsed + 1  
        log.info('parsed {} notes so far'.format(parsed))  
        #signal the end of a piece with 4 long rests
        normalized_notes.extend(('r4','r4','r4','r4'))
        return normalized_notes

    def _stack_approach(self, notes):
        output_notes = []
        prev_length = 0
        prev_offset=0
        rest_dict = {'r1':.25,'r2':.5,'r4':1}
        prev_item = None
        note_list = []
        for item in notes:
           if 'r' in item:
               if prev_item == 'n':
                   note_list,prev_length, prev_offset = self._append_chord_stack(note_list,output_notes, prev_length, prev_offset)
                   new_rest = note.Rest()
                   new_rest.offset = prev_offset+prev_length
                   new_rest.quarterLength = rest_dict[item]
                   prev_length = new_rest.quarterLength
                   prev_offset = new_rest.offset
                   prev_item = 'r'
                   output_notes.append(new_rest)
           elif item.isdigit():
               if prev_item == 'n':
                   note_list, prev_length, prev_offset = self._append_chord_stack(note_list,output_notes, prev_length, prev_offset)
               note_list.append(int(item))                  
               prev_item='n'
           else: #item is stack instruction
               if prev_item == 'n':
                   prev_item = 's'