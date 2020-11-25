import json
import argparse
import glob
import pickle
from music_lstm_generator.utils.model import MidiLSTM
from music_lstm_generator.utils.parser import MidiFileParser


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_weights_path', help='path to weights')
    parser.add_argument('model_params_path', help='path to model params')
    parser.add_argument('parsed_notes_path', help='path to trained notes')
    parser.add_argument('output_path', help='output path for midi')
    parser.add_argument('--primer_sequence', default=None, help='sequence to prime prediction output')
    args = parser.parse_args()
    return args


def load_notes(parsed_notes_path):
    with open(parsed_notes_path, 'rb') as fp:
        notes = pickle.load(fp)
    return notes


def load_model_params(model_params_path):
    with open(model_params_path, "r") as fp:
        model_params_dict = json.loads(model_params_path)
    return model_params_dict


def main():
    args = get_args()

    notes = load_notes(args.parsed_notes_path)
    model_params_dict = load_model_params(args.model_params_path)
    lstm = MidiLSTM(**model_params_dict)
    dim, element_to_int, int_to_element = lstm.build_training_sequences(notes)

    prediction_output = lstm.predict(lstm.model(dim), element_to_int, int_to_element, primer_seq=args.primer_sequence)   
    MidiFileParser(args.output_path).create_pred_midi(prediction_output, args.output_path)

if __name__ == '__main__':
    main()
