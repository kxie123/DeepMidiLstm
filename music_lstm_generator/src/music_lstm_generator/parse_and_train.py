import argparse
import glob
import pickle
from music_lstm_generator.utils.model import MidiLSTM
from music_lstm_generator.utils.parser import MidiFileParser, KEYS

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir')
    parser.add_argument('parsed_notes_dir')
    parser.add_argument('--parse_approach', default = "intval", help='method to parse notes')
    parser.add_argument('--prediction_sequence_length', default=500)
    parser.add_argument('--seq_length', default=60, help='trainig seq length')
    parser.add_argument('--layer_units', default=512)
    parser.add_argument('--dropout_factor', default=0.5)
    parser.add_argument('--learn_rate', default=0.05)
    parser.add_argument('--rho', default=0.9)
    parser.add_argument('--learn_decay', default=0)
    parser.add_argument('--validation_split', default=0.05)
    parser.add_argument('--train_epochs', default=100)
    parser.add_argument('--train_batch_size', default=128)
    parser.add_argument('--init_epoch', default=0)

    args = parser.parse_args()
    return args


def get_notes(train_dir, parse_approach):
    notes = []
    for fp in glob.glob(train_dir+"*.mid"):
        midi_parser = MidiFileParser(fp)
        transpositions = []
        for key in KEYS:
            transpositions.append(midi_parser.transpose_key(key))
        for transposition in transpositions:
            raw_notes_stream = midi_parser.stream_raw_notes(transposition)
            norm_notes = midi_parser.get_normalized_notes(raw_notes_stream, parse_approach)
            notes.extend(norm_notes)
    return notes
            

def main():
    args = get_args()
    #parse and store notes
    notes = get_notes(args.train_dir, args.parse_approach)
    with open(args.parsed_notes_dir+"/notes", 'wb') as out:
        pickle.dump(notes, out)
    #build and train train model
    lstm = MidiLSTM(units=args.units, dropout_factor=args.dropout_factor, lr=args.learn_rate, rho=args.rho, decay=args.decay,
                    validation_split=args.validation_split, epochs=args.epochs, batch_size=args.batch_size, initial_epoch=args.init_epoch)
    dim, encoded_input, encoded_output = lstm.build_training_sequences(notes, args.sequence_length, 1)
    model = lstm.model(dim)

    lstm.train(model, encoded_input, encoded_output)

if __name__ == "__main__":
    main()
