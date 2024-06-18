import os
import numpy as np
import json
import collections
import data_processing as dp
import tokenization
import subprocess
import argparse
import time
from bert_utils import TritonBertClient

# class BertDataReader(CalibrationDataReader):
class BertDataReader():
    def __init__(self,
                 squad_json,
                 vocab_file,
                 batch_size,
                 max_seq_length,
                 doc_stride,
                 start_index=0,
                 end_index=0):
        self.data = dp.read_squad_json(squad_json)
        self.max_seq_length = 384 #max_seq_length
        self.batch_size = batch_size
        self.example_stride = batch_size # number of examples as one example stride. (set to equal to batch size) 
        self.start_index = start_index # squad example index to start with
        self.end_index = len(self.data) if end_index == 0 else end_index 
        self.current_example_index = start_index
        self.current_feature_index = 0 # global feature index (one example can have more than one feature) 
        self.tokenizer = tokenization.BertTokenizer(vocab_file=vocab_file, do_lower_case=True)
        self.doc_stride = doc_stride 
        self.max_query_length = 64
        self.enum_data_dicts = iter([])
        self.features_list = []
        self.token_list = []
        self.example_id_list = []
        self.start_of_new_stride = False # flag to inform that it's a start of new example stride

    def get_next(self, verbose=False):
        iter_data = next(self.enum_data_dicts, None)
        if iter_data:
            self.start_of_new_stride= False
            return iter_data

        self.enum_data_dicts = None
        if self.current_example_index >= self.end_index:
            print("Reading dataset is done. Total examples is {:}".format(self.end_index-self.start_index))
            return None
        elif self.current_example_index + self.example_stride > self.end_index:
            self.example_stride = self.end_index - self.current_example_index

        if self.current_example_index % 10 == 0:
            current_batch = int(self.current_feature_index / self.batch_size) 
            if verbose:
                print("Reading example index {:}, batch {:}, containing {:} sentences".format(self.current_example_index, current_batch, self.batch_size))

        # example could have more than one feature
        # we collect all the features of examples and process them in one example stride
        features_in_current_stride = []
        for i in range(self.example_stride):
            example = self.data[self.current_example_index+ i]
            features = dp.convert_example_to_features(example.doc_tokens, example.question_text, self.tokenizer, self.max_seq_length, self.doc_stride, self.max_query_length)
            self.example_id_list.append(example.id)
            self.features_list.append(features)
            self.token_list.append(example.doc_tokens)
            features_in_current_stride += features
        self.current_example_index += self.example_stride
        self.current_feature_index+= len(features_in_current_stride)


        # following layout shows three examples as example stride with batch size 2:
        # 
        # start of new example stride 
        # |
        # |
        # v
        # <--------------------- batch size 2 ---------------------->
        # |...example n, feature 1....||...example n, feature 2.....| 
        # |...example n, feature 3....||...example n+1, feature 1...| 
        # |...example n+1, feature 2..||...example n+1, feature 3...|
        # |...example n+1, feature 4..||...example n+2, feature 1...|

        data = []
        for feature_idx in range(0, len(features_in_current_stride), self.batch_size):
            input_ids = []
            input_mask = []
            segment_ids = []

            for i in range(self.batch_size):
                if feature_idx + i >= len(features_in_current_stride):
                    break
                feature = features_in_current_stride[feature_idx + i]
                if len(input_ids) and len(segment_ids) and len(input_mask):
                    input_ids = np.vstack([input_ids, feature.input_ids])
                    input_mask = np.vstack([input_mask, feature.input_mask])
                    segment_ids = np.vstack([segment_ids, feature.segment_ids])
                else:
                    input_ids = np.expand_dims(feature.input_ids, axis=0)
                    input_mask = np.expand_dims(feature.input_mask, axis=0)
                    segment_ids = np.expand_dims(feature.segment_ids, axis=0)

            if FLAGS.version == 1.1:
                data.append({"input_ids": input_ids, "input_mask": input_mask, "segment_ids":segment_ids})
            elif FLAGS.version == 2.0:
                data.append({"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids":segment_ids})

        self.enum_data_dicts = iter(data)
        self.start_of_new_stride = True
        return next(self.enum_data_dicts, None)

def get_predictions(example_id_in_current_stride,
                    features_in_current_stride,
                    token_list_in_current_stride,
                    batch_size,
                    outputs,
                    _NetworkOutput,
                    all_predictions):
                    
    if example_id_in_current_stride == []:
        return 

    # print(f"## outputs: {outputs} type: {type(outputs)}")

    base_feature_idx = 0
    for idx, id in enumerate(example_id_in_current_stride):
        features = features_in_current_stride[idx]
        doc_tokens = token_list_in_current_stride[idx]
        networkOutputs = []
        for i in range(len(features)):
            x = (base_feature_idx + i) // batch_size
            y = (base_feature_idx + i) % batch_size

            start_logits = outputs[x].as_numpy(name='output_start_logits')[0]
            end_logits = outputs[x].as_numpy(name='output_end_logits')[0]
            # print(f"## start_logits:\n{start_logits}")
            # print(f"## end_logits:\n{end_logits}")

            networkOutputs.append(_NetworkOutput(
                start_logits = start_logits,
                end_logits = end_logits,
                feature_index = i 
                ))

        base_feature_idx += len(features) 

        # Total number of n-best predictions to generate in the nbest_predictions.json output file
        n_best_size = 20

        # The maximum length of an answer that can be generated. This is needed
        # because the start and end predictions are not conditioned on one another
        max_answer_length = 30

        prediction, nbest_json, scores_diff_json = dp.get_predictions(doc_tokens, features,
                networkOutputs, n_best_size, max_answer_length)

        all_predictions[id] = prediction

def inference(data_reader, client, latency, FLAGS):

    _NetworkOutput = collections.namedtuple(  # pylint: disable=invalid-name
            "NetworkOutput",
            ["start_logits", "end_logits", "feature_index"])
    all_predictions = collections.OrderedDict()
    
    example_id_in_current_stride = [] 
    features_in_current_stride = []  
    token_list_in_current_stride = []
    outputs = []
    while True:
        inputs = data_reader.get_next(FLAGS.verbose)
        if not inputs:
            break

        if data_reader.start_of_new_stride:
            get_predictions(example_id_in_current_stride, features_in_current_stride, token_list_in_current_stride, data_reader.batch_size, outputs, _NetworkOutput, all_predictions)

            # reset current example stride
            example_id_in_current_stride = data_reader.example_id_list[-data_reader.example_stride:]
            features_in_current_stride = data_reader.features_list[-data_reader.example_stride:] 
            token_list_in_current_stride = data_reader.token_list[-data_reader.example_stride:]
            outputs = []

        start = time.time()
        #output = ort_session.run(["output_start_logits","output_end_logits"], inputs)
        output = client.infer(FLAGS, inputs)
        latency.append(time.time() - start)
        outputs.append(output)

    # handle the last example stride
    get_predictions(example_id_in_current_stride, features_in_current_stride, token_list_in_current_stride, data_reader.batch_size, outputs, _NetworkOutput, all_predictions)

    return all_predictions

def get_op_nodes_not_followed_by_specific_op(model, op1, op2):
    op1_nodes = []
    op2_nodes = []
    selected_op1_nodes = []
    not_selected_op1_nodes = []

    for node in model.graph.node:
        if node.op_type == op1:
            op1_nodes.append(node)
        if node.op_type == op2:
            op2_nodes.append(node)

    for op1_node in op1_nodes:
        for op2_node in op2_nodes:
            if op1_node.output == op2_node.input:
                selected_op1_nodes.append(op1_node.name)
        if op1_node.name not in selected_op1_nodes:
            not_selected_op1_nodes.append(op1_node.name)

    return not_selected_op1_nodes


def parse_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", "--model-name", type=str, required=True, help="Name of model"
    )

    parser.add_argument(
        "--version",
        required=False,
        default=1.1,
        help='Squad dataset version. Default is 1.1. Choices are 1.1 and 2.0',
        type=float
    )

    parser.add_argument(
        "--no_eval",
        action="store_true",
        required=False,
        default=False,
        help='Turn off evaluate output result for f1 and exact match score. Default False',
    )

    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        required=False,
        default=1,
        help="Batch size. Default is 1.",
    )

    parser.add_argument("--seq_len",
                        required=False,
                        default=384,
                        help='sequence length of the model. Default is 384',
                        type=int)

    parser.add_argument("--doc_stride",
                        required=False,
                        default=128,
                        help='document stride of the model. Default is 128',
                        type=int)

    parser.add_argument("--samples",
                        required=False,
                        default=0,
                        help='Number of samples to test with. Default is 0 (All the samples in the dataset)',
                        type=int)


    parser.add_argument(
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help='Show verbose output',
    )

    parser.add_argument(
        "-i",
        "--protocol",
        type=str,
        required=False,
        default="HTTP",
        help="Protocol (HTTP/gRPC) used to communicate with "
        + "the inference service. Default is HTTP.",
    )

    parser.add_argument(
        "image_filename",
        type=str,
        nargs="?",
        default=None,
        help="Input image / Input folder.",
    )

    parser.add_argument(
        "--streaming",
        action="store_true",
        required=False,
        default=False,
        help="Use streaming inference API. "
        + "The flag is only available with gRPC protocol.",
    )

    parser.add_argument(
        "-a",
        "--async",
        dest="async_set",
        action="store_true",
        required=False,
        default=False,
        help="Use asynchronous inference API",
    )

    parser.add_argument(
        "-x",
        "--model-version",
        type=str,
        required=False,
        default="",
        help="Version of model. Default is to use latest version.",
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    '''
    BERT QDQ Quantization for MIGraphX.

    There are two steps for the quantization,
    first, calibration is done based on SQuAD dataset to get dynamic range of floating point tensors in the model
    second, Q/DQ nodes with dynamic range (scale and zero-point) are inserted to the model

    The onnx model used in the script is converted from MLPerf,
    https://zenodo.org/records/3733910
    
    Some utility functions for dataset processing, data reader and evaluation are from Nvidia TensorRT demo BERT repo,
    https://github.com/NVIDIA/TensorRT/tree/master/demo/BERT
    
    '''

    FLAGS = parse_input_args()

    if FLAGS.samples and FLAGS.samples < FLAGS.batch_size:
        print("Error: Sample count must be batch size or larger. Exiting")
        exit()
    
    # Set squad version   
    if FLAGS.version == 1.1:
        squad_json = "./squad/dev-v1.1.json"
    elif FLAGS.version == 2.0:
        squad_json = "./squad/dev-v2.0.json" # uncomment it if you want to use squad v2.0 as dataset
    else:
        print("Error: version " + str(FLAGS.version) + " of squad dataset not a valid choice")
        exit()
    
    vocab_file = "./squad/vocab.txt"
    # augmented_model_path = "./augmented_model.onnx"

    sequence_lengths = [FLAGS.seq_len] # if use sequence length 384 then choose doc stride 128. if use sequence length 128 then choose doc stride 32. 
    

    doc_stride = [FLAGS.doc_stride]
    batch_size = FLAGS.batch_size 

    data_reader = BertDataReader(squad_json, vocab_file, batch_size, sequence_lengths[-1], doc_stride[-1], end_index=FLAGS.samples)
    
    # create triton bert client
    client = TritonBertClient(FLAGS)
    print("Running Inferences")
    latency = [] #Used for timing information
    all_predictions = inference(data_reader, client, latency, FLAGS) 

    print("Inference Complete!")
    print(f"{latency =}")
    print("Rate = {} QPS ".format(
        format((((FLAGS.batch_size) / (sum(latency[1:]) / len(latency[1:])))),
                '.2f')))
    print("Average Execution time = {} ms".format(
            format(sum(latency[1:]) * 1000 / len(latency[1:]), '.2f')))

    # Verify output result from run
    if not FLAGS.no_eval:
        print(" Saving predictions")
        prediction_file = "./prediction.json"
        with open(prediction_file, "w") as f:
            f.write(json.dumps(all_predictions, indent=4))
            print("\nOutput dump to {}".format(prediction_file))


        print("Evaluate QDQ model for SQUAD v"+ str(FLAGS.version))
        subprocess.call(['python3', './squad/evaluate-v'+ str(FLAGS.version) + '.py', './squad/dev-v' + str(FLAGS.version) + '.json', './prediction.json', '90'])

