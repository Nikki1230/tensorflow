from utils import *
from network import *

def test(model_path, test_data,vocab_size,id_to_word):
	test_input = Input(batch_size=20,num_steps=35,data=test_data)

	m = Model(test_input, is_training=False,hidden_size=650,vocab_size=vocab_size, num_layer=2)

	saver= tf.train.Saver()

	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		threads = tf.train().start_queue_runners(coord=coord)

		current_state = np.zeros((2, 2, m.batch_size,m.hidden_size))
		saver.restore(sess,model_path)

		num_acc_batches =30

		check_batch_idx = 25

		acc_check_thresh =5
		
		accuracy = 0
		for batch in range(num_acc_natches):
			if batch == check_batch_idx:
				true, pred, current_state,acc = sess.run([m.input_obj.targets,m.predict,m.stat,m.accuracy],
									feed_dict={m.init_state:current_state})

				pred_words = [id_to_words[x] for x in pred[:m.num_steps]]
				true_words = [id_to_words[x] for x in true[0]]

				print("True words (1st line) VS predicted words (2nd line):")
				print("".join(true_words))
				print("".join(pred_words))
			else:
				acc, current_state = sess.run([m.accuracy,m.state],feed_dict={m.init_state:current_state})

			if batch >= acc_check_thresh:
				accuracy +=acc

		print("Average accuracy :{:.3f}".format(accuracy / (num_acc_batches - acc_check_thresh)))



		coord.request_stop()
		coord.join(threads)

if __name__="__main__":
	if args.data_path:
		data_path= args.data_path
	if args.load_file:
		load_file = args.load_file
	train_data, valid_data, test_data,vocab_size,id_to_word=load_data(data_path)

	trained_model = save_path + '/'+load_file

	test(trained_model,test_data, vocab_size, id_to_word)
		
