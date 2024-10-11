# MEASG
MEASG： Multi-Encoder and Abstract Syntax Graph for Code Summarization 
# Requirements
pytorch>=1.4.1 \
tqdm \
nltk \
torch_geometric
### The following code should be run in advance.
import nltk \
nltk.download('wordnet')

# Dataset
Using the funcom dataset, the following link follows:
https://s3.us-east-2.amazonaws.com/icse2018/index.html

# Dataset processing
## Data Processing Steps
### *1. loadData.py - Extract Source Code and Comments from the Original Funcom Source Files.
- **Outputs**: `source_code1.pkl`, `source_comment1.pkl`
### *2. gen_java_files.py - Convert Source Code from `source_code1.pkl` into Individual Java Files for AST Conversion.
- **Input**: `source_code1_small.pkl`
- **Output**: Individual Java files
- **Details**: 
    - Total Java files: 4,298,242
    - After removing half of the log files: 2,149,121 Java files
### *3. Run Command Line to Convert Java Files to AST
- **Command**: `java -jar asts.jar -outdir ../java_dots -ast -lang java -format dot -node_level block ../java_files`
- **Output**: AST `.dot` files saved in `E:\Repetition_Experiment\SourceCode\SSMIF_Data\small_Funcom\java_dots`
- **Details**:
    - Total DOT files: 4,271,064
    - After removing half of the log files: 2,135,532 DOT files
### *4. get_correct_splitted_ast.py - Obtain the AST IDs for the Training, Testing, and Validation Sets, Excluding Code-Comment Pairs Without ASTs.
- **Input**: `source_code1_small.pkl`
- **Output**: `correct_splitted_ast_fid.pkl` (2,121,369 entries)
### *5. get_splitted_ast.py - Extract AST Trees, Where Each Subtree Is a Separate List, and Each Node Forms a List.
- **Input**: `correct_splitted_ast_fid.pkl`
- **Outputs**: `sliced_AST.pkl`, `correct_parsed_splitted_ast_fid.pkl`, `special_Node_total.pkl`
### *6. get_ast_input.py - Convert All AST Words to Lowercase, Replace Numbers and Strings, and Generate Vocabulary Input.
- **Note**: No camelCase splitting is applied.
- **Input**: `sliced_AST.pkl`
- **Outputs**: `sliced_flatten_ast.pkl`, `sliced_flatten_ast.pkl`, `ast_vocab_input.pkl`
### *7. humpCut_code.py - Apply CamelCase Splitting Principles on the Source Code, Storing the Result as a Dictionary (Key: Source Code ID). Non-English characters are separated by spaces, based on the IDs in `correct_parsed_splitted_ast_fid.pkl`.
- **Input**: `source_code1_small.pkl`, `sliced_flatten_ast_fid.pkl`
- **Outputs**: `token_source.pkl`, `token_code.pkl`
### *8. getDegreeList.py - Convert Edge Relationships into Two Lists: Starting Node List and Ending Node List.
- **Input**: `sliced_flatten_ast.pkl`
- **Output**: `sliced_flatten_ast_degreeList.pkl`




# Config
# Configuration Parameters

| Parameter Name          | Description                              | Default Value     |
|-------------------------|------------------------------------------|-------------------|
| `debug`                 | Whether to enable debug mode             | `True`            |
| `selectData`            | Dataset selection                        | `"Funcom"`        |
| `data_root_path`        | Root path for the data                   | `"../Data/Funcom"`|
| `correct_fid`           | Path for correct file IDs                | `data_root_path/correct_fids` |
| `dots_files_dir`        | Path for Java DOT files                  | `"../../SSMIF_Data/Funcom/java_dots"` |
| `code_summary_dir`      | Path for code summaries                  | `data_root_path/source_code2summary` |
| `token_dir`             | Path for code tokens                     | `data_root_path/token_source_code` |
| `ASTs_dir`              | Path for ASTs                            | `data_root_path/ASTs` |
| `GNNData_dir`           | Path for GNN data                        | `data_root_path/GNNData` |
| `plt_photo_dir`         | Path for plots/photos                    | `"../plt_photo/"` |
| `output_root`           | Root path for outputs                    | `../output/Funcom/<timestamp>` |
| `model_root`            | Path for saving models                   | `output_root/models` |
| `vocab_root`            | Path for saving vocabularies             | `output_root/vocab` |
| `DDP`                   | Use Distributed Data Parallel (DDP)      | `False`           |
| `batch_size`            | Training batch size                      | `64`              |
| `valid_batch_size`      | Validation batch size                    | `128`             |
| `test_batch_size`       | Test batch size                          | `128`             |
| `learning_rate`         | Learning rate                            | `0.000055`        |
| `lr_decay_rate`         | Learning rate decay rate                 | `0.9`             |
| `embedding_dim`         | Embedding layer dimension                | `256`             |
| `hidden_size`           | Hidden layer size                        | `256`             |
| `decoder_dropout_rate`  | Dropout rate for the decoder             | `0.5`             |
| `n_epochs`              | Number of training epochs                | `50`              |
| `teacher_forcing_ratio` | Teacher forcing ratio                    | `0.5`             |
| `use_cuda`              | Use CUDA acceleration                    | `torch.cuda.is_available()` |
| `device`                | Device configuration                     | `torch.device('cuda:0')` |
| `init_uniform_mag`      | Magnitude for uniform initialization     | `0.02`            |
| `init_normal_std`       | Standard deviation for normal initialization | `1e-4`          |
| `eps`                   | Small constant to avoid division by zero | `1e-12`           |
| `continue_train`        | Continue training                        | `True`            |
| `max_ast_length`        | Maximum AST length                       | `2000`            |
| `max_subast_len`        | Maximum sub-AST length                   | `100`             |
| `max_code_length`       | Maximum code length                      | `200`             |
| `max_nl_length`         | Maximum natural language length          | `30`              |
| `min_nl_length`         | Minimum natural language length          | `4`               |
| `max_decode_steps`      | Maximum decoding steps                   | `30`              |
| `early_stopping_patience`| Early stopping patience                 | `20`              |
| `source_vocab_size`     | Vocabulary size for source code          | `50000`           |
| `code_vocab_size`       | Vocabulary size for code tokens          | `50000`           |
| `ast_vocab_size`        | Vocabulary size for ASTs                 | `50000`           |
| `nl_vocab_size`         | Vocabulary size for natural language     | `50000`           |
| `use_pointer_gen`       | Use pointer generation mechanism         | `True`            |
| `use_teacher_forcing`   | Use teacher forcing                      | `True`            |
| `use_lr_decay`          | Use learning rate decay                  | `True`            |
| `use_early_stopping`    | Use early stopping mechanism             | `True`            |
| `save_valid_model`      | Save model on validation                 | `True`            |
| `save_best_model`       | Save the best model                      | `True`            |
| `save_test_outputs`     | Save test outputs                        | `True`            |
| `beam_width`            | Beam search width                        | `5`               |
| `beam_top_sentences`    | Number of top sentences from beam search | `1`               |
| `log_state_every`       | Log state every X steps                  | `1000`            |


# Train

python main.py




# Eval 
cd EvaluationMetrics \
python MEASG_score.py



# Model
![image](https://github.com/AFunFun222/MEASG/tree/main/pic/1.png)
