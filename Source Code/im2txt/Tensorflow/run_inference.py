import vocabulary
import inference_wrapper
import configuration
import h5py
import tensorflow as tf
import caption_generator
import math

vocab = vocabulary.Vocabulary("data/dictionary.txt")

print(vocab.start_id)

print(vocab.end_id)

print(vocab.unk_id)

model = inference_wrapper.InferenceWrapper()
restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               "train_log/393000.ckpt")

file = h5py.File("data/image_vgg19_fc1_feature.h5", 'r')
encoded_images = file['train_set']

sess = tf.InteractiveSession()
restore_fn(sess)



generator = caption_generator.CaptionGenerator(model, vocab)


for index in range(30):
    captions = generator.beam_search(sess, encoded_images[index])
    print("Captions for image {}".format(index + 1))
    for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w - 1) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))