import vocabulary
import inference_wrapper
import configuration
import h5py
import tensorflow as tf
import caption_generator
import math

vocab = vocabulary.Vocabulary("data/segment/dictionary.txt")
file = h5py.File("data/image_vgg19_fc1_feature.h5", 'r')
encoded_images = file['validation_set']

check_point_steps = 393000

model = inference_wrapper.InferenceWrapper()
restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               "check_point_files/segment/393000_steps/{}.ckpt".format(check_point_steps))

sess = tf.InteractiveSession()
restore_fn(sess)

generator = caption_generator.CaptionGenerator(model, vocab)

# output three optional sentences for each image, ranking by probability in decreasing order
# with open('train_caption_output.txt', 'w') as f:
#     for iter in range(400):
#         index = 2 + iter * 20
#         captions = generator.beam_search(sess, encoded_images[index])
#         f.write("Captions for image {}\r\n".format(index + 1))
#         print("Captions for image {}".format(index + 1))
#         for i, caption in enumerate(captions):
#             # Ignore begin and end words.
#             sentence = [vocab.id_to_word(w - 1) for w in caption.sentence[1:-1]]
#             sentence = " ".join(sentence)
#             f.write("  %d) %s (p=%f)\r\n" % (i, sentence, math.exp(caption.logprob)))
#             print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

# output inferencing result following requirements in the submitting web page
with open('D:\Image-Caption\Results\\validation_output\\valid_caption_output_segment_{}_emb_512.txt'.format(check_point_steps),
            'w') as f:
    for index in range(1000):
        captions = generator.beam_search(sess, encoded_images[index])
        caption = captions[0]
        sentence = [vocab.id_to_word(w - 1) for w in caption.sentence[1:-1]]
        full_str = "".join(sentence)
        f.write(repr(8000 + index))
        for word in full_str:
            f.write(' {}'.format(word))
        f.write('\n')
        if (index + 1) % 100 == 0:
            print(repr(index + 1))