import optparse


def standard_parser():
    parser = optparse.OptionParser()

    # Option parser
    parser.add_option('-c', '--comment',
                      action="store", dest="comment",
                      help="Comment about the model", type="string", default="")

    parser.add_option('-m', '--config',
                      action="store", dest="config",
                      help="Path to a config file", type="string",
                      default="/Users/kforest/workspace/toxiccomment/models/keras_config.json")

    options, args = parser.parse_args()

    return options, args
