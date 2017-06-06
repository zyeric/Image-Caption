import vocabulary
import inference_wrapper
import configuration
import h5py
import tensorflow as tf
import caption_generator
import math

vocab = vocabulary.Vocabulary("data/dictionary.txt")

model = inference_wrapper.InferenceWrapper()
restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               "train_log/12000.ckpt")

file = h5py.File("data/image_vgg19_fc1_feature.h5", 'r')
encoded_images = file['train_set']

sess = tf.InteractiveSession()
restore_fn(sess)



generator = caption_generator.CaptionGenerator(model, vocab)


choose_index = [3, 23, 43, 63, 83]

for index in choose_index:
    captions = generator.beam_search(sess, encoded_images[index-1])
    print("Captions for image {}".format(index))
    for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))