### Using Music LSTM generator 

The music LSTM generator is an application that allows for training an RNN LSTM to generate music. The application parses a collection of midi files, trains a model to your specifications and generates a midi file composition in the style of the music you trained it on!

### Sample results

https://soundcloud.com/kenny-xie-810483223/sets/music-lstm

### How to use it

To train a model:
1. Specify a directory to use for the training music and storing the parsed notes
2. Run python3 `parse_and_train.py [train_dir] [parsed_notes_dir] ...[optional model param args]...`

To generate predictoins:
3. Run python3 `generate_music.py [model_weights_path] [model_params_path] [parsed_notes_path] [output_path] [primer_sequence]`


### Background and inspiration

I studied music in college and have played violin my whole life. In college, I majored in statistics, computer science and minored in music. In the Spring of 2018, I took an NLP and text mining class, where I learned about semantic relationships and text generation techniques. An idea suddenly came to me - wouldn't it be cool if I used my CS skills to create an application to generate original music? After all, music is the universal language. In the same way characters and words are treated, notes and chords can be parsed in similar ways to represent the unique problem space of music lexicon. With this thought, I set a goal during the summer of 2018 to create an application that could learn the language of music and generate melodies and chords with at least a rudimentary sense of tonality. 

### Parsing

Perhaps the most intersting part of the process in building this application was not the model itself (built using existing frameowrks), but rather figuring out how to parse and feed the LSTM with sequences of notes and instructions. 

* Individual notes: This is probably the most straight forward step. Notes can be represented by their **midi value**, a value from 0-127 across 8 octaves. 
* Chords: A chord can be represented by simply a stack of notes, such as '60-64-67', which represents a c major chord. However, this significantly increases the dimentionality of the problem space since there are so many distinct combinations of chords. The approach I eventually decided on was to use "-" as an instruction and the intervals, such as "4" and "7" in this case to sequentially build the chord. This way, the problem space is simplified and the neural network can also learn generally the relative intervals for chords. 
* Rests and time: For simplicity, all note values are normalized, because the LSTM by nature is one dimensional. Rests are thus used instead to represent rhythm and note length. Rests are normalized to values of 'r1'..'r4' for lengths of quarter rest to full rest. 

* Parsing the score: For simplicity again, time signatures accepted are in 2 or in 4. The midi scores are also "flattened" to ensure we are only considering "one voice" or the melody. Scores with counterpoint and multive voices are particularly difficult for an LSTM to train on and generate because of the increased complexity of musical relationships. 

* Transpositions: Transpositions are interesting because the relative relationships between notes are all the same, but to an LSTM with one-hot encoded sequences, each transposition of notes is a new set of vocabulary and relationships to learn. Thus, when parsing every piece is transposed into all 12 keys to represent a complete set of relationships in every key. 

Sample parsed text sequence of notes: `...72r177r179r181-12-40r2r181r177-12-32r2r172r177-12-29r179r181-12-40r2r181r145r184r2r184-12-36r2r181-12-33r179r2r152r4r155r2r148r4r152r4r179-12-24r181r182-12-34r182r182r182-12-30r2r179r179-12-24r181r182-12-34r2r182r184-12-32r2r182r181-12-26r179r181-12-40r4r145r4...`

### Model

The model is a standard LSTM RNN built using Keras framework which includes dropout layers to prevent overfitting in music and categorical cross entropy as loss function (since our vocab is of distinct classes). For simplicity, one-hot encoding was used to encode sequences of notes and instructions. A lot of credit goes to: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py from where I referenced the general methodology of building sequences for training. 

I also am using a temperature scaling factor to increase the randomness of predictions before the softmax is applied. This helps prevent repetitive generation. All the hyperparametesr can be tuned in the application. 

In the pre_trained_models directory, you are able to see a few models and the epochs trained for each model.
