
def inference(args, input_words, pre_motion_seq, encoder, gen, lang_model):
    input_length = [len(input_words) + 2]  # +2 for SOS and EOS
    input_seq = np.zeros((input_length[0], 1))  # [seq x batch]
    input_seq[0, 0] = lang_model.SOS_token
    for i, word in enumerate(input_words):
        try:
            word_idx = lang_model.word2index[word]
        except KeyError:
            word_idx = lang_model.UNK_token
        input_seq[i + 1, 0] = word_idx
    input_seq[input_seq.shape[0] - 1, 0] = lang_model.EOS_token
    input_seq = torch.from_numpy(input_seq).long().to(device)
    pre_motion_seq = torch.from_numpy(pre_motion_seq)  # [seq x dim]

    # encoding
    encoder_outputs, encoder_hidden = encoder(input_seq, input_length, None)
    gen_hidden = encoder_hidden[:gen.n_layers]

    target_length = args.pre_motion_steps + args.estimation_steps
    motion_output = np.array([])
    attentions = torch.zeros(target_length, len(input_seq))

    # time steps
    for t in range(target_length):
        if t < args.pre_motion_steps:
            gen_input = pre_motion_seq[t].unsqueeze(0).to(device).float()
            gen_output, gen_hidden, attn_weights = gen(gen_input, gen_hidden, encoder_outputs)
        else:
            gen_input = gen_output
            gen_output, gen_hidden, attn_weights = gen(gen_input, gen_hidden, encoder_outputs)

            if t == args.pre_motion_steps:
                motion_output = gen_output.data.cpu().numpy()
            else:
                motion_output = np.vstack((motion_output, gen_output.data.cpu().numpy()))

        if attn_weights is not None:
            attentions[t] = attn_weights.data

    return motion_output, attentions


def main(mode):
    checkpoint_path = '../saved_model_1/seq2seq_checkpoint_160.bin'
    args, encoder, gen, lang_model, out_dim, data_norm_stats = load_checkpoint(checkpoint_path)
    pprint.pprint(vars(args))

    # load pca
    pca = None
    if args.pose_representation == 'pca':
        with open(args.pose_representation_path, 'rb') as f:
            pca = pickle.load(f)

    def infer_from_words(words, duration=None):
        """ infer speech gestures given speech text and length in seconds """
        start = time.time()

        if duration is None:
            duration = len(words) / 2.5  # assume average speech speed (150 wpm = 2.5 wps)

        unit_duration = 0.08333  # seconds per frame (the dataset has 12 fps)
        pre_duration = args.pre_motion_steps * unit_duration
        motion_duration = args.estimation_steps * unit_duration
        num_words_for_pre_motion = round(len(words) * pre_duration / duration)
        num_words_for_estimation = round(len(words) * motion_duration / duration)

        # pad some dummy words for the first chunk
        padded_words = ['<UNK>'] * num_words_for_pre_motion + words

        # split chunks and inference
        InferenceOutput = namedtuple('InferenceOutput', ['words', 'pre_motion_seq', 'out_motion', 'attention'])
        pre_motion_seq = np.zeros((args.pre_motion_steps, out_dim))
        outputs = []

        for i in range(0, len(padded_words) - num_words_for_pre_motion, num_words_for_estimation):
            sample_words = padded_words[i:i + num_words_for_pre_motion + num_words_for_estimation]
            with torch.no_grad():
                output, attention = inference(args, sample_words, pre_motion_seq,
                                              encoder=encoder, gen=gen, lang_model=lang_model)

            outputs.append(InferenceOutput(sample_words, pre_motion_seq, output, attention))

            # prepare next posture input
            pre_motion_seq = np.asarray(output)[-args.pre_motion_steps:, :]

        print('inference took {:0.2f} seconds'.format(time.time() - start))

        return outputs

    # inference
    sentence = "look at the big world in front of you ,"

    words = normalize_string(sentence).split(' ')
    outputs = infer_from_words(words)


