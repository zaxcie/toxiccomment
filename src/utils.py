import optparse


def get_standard_parser():
    parser = optparse.OptionParser()

    # TODO implement comment text file

    # Option parser
    parser.add_option('-c', '--comment',
                      action="store", dest="comment",
                      help="Comment about the model", type="string", default="")

    parser.add_option('-m', '--modeldir',
                      action="store", dest="model_dir",
                      help="Path to store model", type="string",
                      default="/Users/kforest/Documents/workspace/toxiccomment/models")

    parser.add_option('-d', '--datadir',
                      action="store", dest="data_dir",
                      help="Path to load data", type="string",
                      default="/home/ubuntu/efs/IsCar/")

    options, args = parser.parse_args()

    return options, args


def write_comment(MODEL_COMMENT, MODEL_DIR, MODEL_NAME):
    # Write comment on disk
    with open(MODEL_DIR + "/comment_logs.txt", "a") as f:
        line = str(str(MODEL_NAME) + " - " + MODEL_COMMENT)
        f.writelines(line)
