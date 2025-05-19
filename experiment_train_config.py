import argparse

# Define the argument parser
parser = argparse.ArgumentParser(description="Train a Knowledge-Based Large Language Model (KB-LLM)")

# Add arguments
parser.add_argument("--seed",
                    type=int,
                    default=1,
                    help="Random seed for reproducibility")

parser.add_argument("--train_dataset",
                    type=str,
                    default="gpt_data",
                    help="Name of the training dataset")

parser.add_argument("--num_train_sample",
                    type=int,
                    default=120000,
                    help="Number of training samples to use (selects the first N)")

parser.add_argument("--batch_size",
                    type=int,
                    default=10,
                    help="Batch size")

parser.add_argument("--total_steps",
                    type=int,
                    default=20000,
                    help="Total training steps")

parser.add_argument("--encoder_specification",
                    type=str,
                    default="OAI",
                    help="Specification of the encoder to use (e.g.,OAI,Sentence - BERT)")

parser.add_argument("--key_embed_source",
                    type=str,
                    default="key",
                    choices=["key", "answer", "questions", None],
                    help="Source of key embeddings (key,answer,questions,or None)")

parser.add_argument("--use_data_augment",
                    action="store_true",
                    help="Use data augmentation (randomly select templates for questions)")

parser.add_argument("--use_learning_rate_decay",
                    action="store_true",
                    help="Use learning rate decay during training")

parser.add_argument("--use_cached_embed",
                    action="store_true",
                    help="Use pre-computed key-value embeddings")

parser.add_argument("--dataset_dir",
                    type=str,
                    default="synthetic_data",
                    help="Directory containing the dataset")

parser.add_argument("--model_dir_to_resume",
                    type=str,
                    default=None,
                    help="Directory of a checkpoint to resume training from")

parser.add_argument("--model_save_dir",
                    type=str,
                    default="output",
                    help="Directory to save model checkpoints")

parser.add_argument("--separate_query_head",
                    action=argparse.BooleanOptionalAction,
                    help="Use a separate query head for the knowledge base")

parser.add_argument("--kb_size",
                    type=int,
                    default=None,
                    help="Fixed size of the knowledge base")

parser.add_argument("--dynamic_kb_size",
                    nargs=2,
                    type=int,
                    default=None,
                    help="Dynamic range for knowledge base size (min max)")

parser.add_argument("--max_seq_len",
                    type=int,
                    default=None,
                    help="Maximum sequence length")

parser.add_argument("--gradient_accumulation_step",
                    type=int,
                    default=20,
                    help="Number of gradient accumulation steps")

parser.add_argument("--length_invariance",
                    action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Scale the raw attention score")

parser.add_argument("--outlier_num",
                    type=int,
                    default=1,
                    help="Number of questions without correct KB entities to include")

parser.add_argument("--multi_entity",
                    type=int,
                    default=None,
                    help="Number of questions involving multiple entities")

parser.add_argument("--use_extended_qa",
                    action="store_true",
                    help="Use extended open-ended QA")

parser.add_argument("--kb_token_layer_frequency",
                    type=int,
                    default=3,
                    help="Frequency of KB token layers")

parser.add_argument("--llm_type",
                    type=str,
                    default="llama3",
                    choices=["llama3",
                             "phi3"],
                    help="Type of LLM to use (llama3 or phi3)")

parser.add_argument("--hf_model_specification",
                    type=str,
                    default="meta-llama/Llama-3.2-1B-Instruct",
                    choices=["meta-llama/Meta-Llama-3-8B",
                             "microsoft/Phi-3-mini-4k-instruct",
                             "meta-llama/Llama-3.2-1B-Instruct"],
                    help="Hugging Face model specification")

parser.add_argument("--hf_token",
                    type=str,
                    default=None,
                    help="Hugging Face API token (required for some models)")

parser.add_argument("--verbose",
                    action="store_true",
                    help="Enable verbose logging (debug level)")

parser.add_argument("--log_to_file",
                    action="store_true",
                    help="Log to file in addition to stdout")

parser.add_argument("--duplicate_true_kb",
                    action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Duplicate true entity's KB token")

parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")




# Function to parse the arguments
def get_arguments():
    return parser.parse_args()
