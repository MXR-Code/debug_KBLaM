import json
import logging
import os
import pathlib

from functools import partial
from itertools import chain
from typing import Callable, Dict, List, Optional

import wandb
import numpy as np

from torch.nn.parallel import DistributedDataParallel

import transformers
from transformers import AutoTokenizer

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn
from torch.nn import CrossEntropyLoss
from accelerate import Accelerator

from model_kblam_kb_encoder import KBEncoder  # 导入知识库编码器模型
from model_kblam_config import KBLaMConfig  # 导入KBLaM配置
from model_llama3 import KblamLlamaForCausalLM  # 导入Llama生成模型
from model_phi3 import KBLaMPhi3ForCausalLM  # 导入Phi模型

from utils_data import augment_row, generate_multi_entity_qa, get_i_dont_know_ans  # 导入数据处理工具
from utils_train import context_set_size_scheduler, get_kb_embd, setup_scheduler_and_optimizer  # 导入训练工具

from model_kblam_kb_retriever import KBRetriever  # 导入知识库检索器
from experiment_train_function_for_llama import *  # 导入Llama训练函数
from experiment_train_function_for_phi import *  # 导入Phi训练函数

from experiment_train_config import get_arguments  # 导入训练配置解析器


def create_custom_progress_bar(console: Console = None,  # type: ignore
                               color: str = "cyan",
                               show_time: bool = True,
                               show_spinner: bool = True,
                               spinner_style: str = "dots",
                               disable=False,
                               ) -> Progress:
    """
    使用Rich创建一个自定义的进度条，选择性地包括损失报告。
    :param description: 任务描述
    :param total: 总步骤数
    :param console: Rich控制台对象（如果为None，将创建一个新的）
    :param color: 进度条颜色
    :param show_time: 是否显示剩余时间
    :param show_spinner: 是否显示旋转图标
    :param spinner_style: 旋转图标样式（例如，"dots"，"dots12"，"line"，"arrow"）
    :param show_loss: 是否显示损失信息
    :return: 一个Rich进度对象和任务ID
    """
    if console is None:
        console = Console()
    columns = []

    if show_spinner:
        columns.append(SpinnerColumn(spinner_name=spinner_style, style=color))

    columns.extend([
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None, style=color, complete_style=f"bold {color}"),
        TaskProgressColumn(),
        TextColumn("[bold yellow]Loss: {task.fields[loss]:.4f}", justify="right"),
    ]
    )

    if show_time:
        columns.append(TimeRemainingColumn())

    progress = Progress(*columns, console=console, expand=True, disable=disable)  # 创建进度条对象
    return progress


def get_batch(qa_format_func: Callable[[str, str], str],  # 获取一个batch数据
              label_func: Callable[[torch.Tensor, List, Callable], torch.Tensor],
              dataset: List[Dict],  # 整个数据集
              tokenizer,
              device: torch.device,
              B: int = 20,  # 默认batch大小为20
              random_sample=True,  # 是否随机抽样
              use_data_aug=False,  # 是否使用数据增强
              include_outlier=False,  # 是否包含异常值
              multi_entities=None,  # 是否使用多重实体
              use_extended_qa=False,  # 是否使用扩展QA
              ):
    """
    dataset: 字典列表，表示知识库，用于提取QA对
    model: LLM，用于提供嵌入
    kb_embedding: KB嵌入（可微分）
    B: 批处理大小
    include_outlier: 生成一个没有答案的batch
    multi_entities: 创建涉及多个实体的批问
    """
    labels = []  # 存储标签
    if multi_entities is not None:
        assert not include_outlier  # 如果多重实体不为None，则不能包含异常值

    if random_sample:  # 如果需要随机抽样
        if multi_entities is not None:
            batch_indices = np.random.choice(len(dataset), (B, multi_entities), replace=False)  # 抽取多重实体的索引
        else:
            batch_indices = np.random.choice(len(dataset), B, replace=False)  # 抽取普通数据的索引
    else:
        batch_indices = np.arange(B)  # 不随机，返回一个范围

    def get_question_and_answer(idx: int) -> tuple[str, str]:  # 根据索引获取问答对
        if use_extended_qa:
            Q, A = dataset[idx]["extended_Q"], dataset[idx]["extended_A"]  # 扩展问答

        elif multi_entities is not None:  # 如果是多重实体
            Q, A = generate_multi_entity_qa([dataset[i]["name"] for i in idx],
                                            [dataset[i]["description_type"] for i in idx],
                                            [dataset[i]["description"] for i in idx],
                                            )
        else:
            Q = augment_row(dataset[idx]) if use_data_aug else dataset[idx]["Q"]  # 数据增强或直接获取问题
            A = get_i_dont_know_ans() if include_outlier else dataset[idx]["A"]  # 异常值或直接获取答案
        return Q, A

    with torch.autograd.no_grad():  # 禁用梯度计算
        input_strs = []  # 存储输入字符串
        real_batch_indices = []  # 存储实际的batch索引
        for idx in batch_indices:
            Q, A = get_question_and_answer(idx)
            if Q is not None and A is not None:  # 确保问题和答案都存在
                input_strs.append(qa_format_func(Q, A))  # 格式化问答对
                real_batch_indices.append(idx)  # 记录有效的索引
            else:
                print("Q or Answer is none")  # 如果问题或答案为空，打印警告
        batch_indices = real_batch_indices
        tokenizer_output = tokenizer(input_strs, return_tensors="pt", padding=True).to(device)  # 使用tokenizer处理输入并发送到设备
        input_ids, attention_masks = (tokenizer_output["input_ids"],
                                      tokenizer_output["attention_mask"],
                                      )

        labels = label_func(input_ids, input_strs, tokenizer)  # 用于标签处理
    if include_outlier:
        # 生成新的索引集，确保知识库不包含问题来源的实体
        batch_indices = np.random.choice(len(dataset), B, replace=False)  # 重新生成随机索引
    return input_ids, attention_masks, labels, batch_indices  # 返回输入ID、注意力掩码、标签和batch索引


def get_prefix_str(args):  # 生成实验前缀字符串
    prefix_string = f"stage1_LearningRate{args.learning_rate}"

    if args.kb_token_layer_frequency is not None:
        prefix_string += f"_KBTokenLayerFreq{args.kb_token_layer_frequency}"

    if args.use_extended_qa:
        prefix_string += "_UseExtendedQA"

    if args.multi_entity is not None:
        prefix_string += f"_MultiEntities{args.multi_entity}"

    if args.outlier_num != -1:
        prefix_string += f"_UseOutlier{args.outlier_num}"

    if args.length_invariance:
        prefix_string += "_LengthInvariant"

    if not args.duplicate_true_kb:
        prefix_string += "_NoDuplicate"

    if args.kb_size is not None:
        prefix_string += f"_KBSize{args.kb_size}"

    if args.dynamic_kb_size is not None:
        prefix_string += "_KBSizeDynamic"

    if args.separate_query_head:
        prefix_string += "_SepQueryHead"

    if args.use_data_augment:
        prefix_string += "_UseDataAug"

    return prefix_string


def load_cached_embeddings(encoder_specification: str, dataset_dir: str, dataset_name: str, key_embed_source: str):
    # 加载缓存的嵌入
    if encoder_specification == "OAI":
        encoder_model_spec_str = "oai"
    else:
        encoder_model_spec_str = encoder_specification
    key_embds = np.load(os.path.join(dataset_dir,
                                     f"{dataset_name}_{encoder_model_spec_str}_embd_{key_embed_source}.npy",
                                     )
                        ).astype("float32")  # 加载键嵌入
    if key_embed_source == "answer":
        # 如果我们使用答案字符串作为键，则也使用它作为值字符串
        value_embds = np.load(os.path.join(dataset_dir,
                                           f"{dataset_name}_{encoder_model_spec_str}_embd_answer.npy",
                                           )
                              ).astype("float32")  # 加载值嵌入
    else:
        value_embds = np.load(os.path.join(dataset_dir,
                                           f"{dataset_name}_{encoder_model_spec_str}_embd_value.npy",
                                           )
                              ).astype("float32")  # 加载备选值嵌入
    return key_embds, value_embds  # 返回键嵌入和值嵌入


def get_step_config(current_accum_step: int,
                    total_accum_step: int,
                    use_data_aug: bool,
                    outlier_num: int,
                    multi_entities: int | None,
                    use_extended_qa: bool,
                    ):
    """
    我们的指令调整数据集由不同类型的指令组成。
    策略：
    异常QA需要最后`outlier_num`累积步骤；
    多重实体QA（如果包含）占其余累积步骤的1/3；
    扩展QA（如果包含）占其余累积步骤的1/3；
    标准QA占其余部分。
    """
    config = {}
    config["use_data_aug"] = use_data_aug
    config["include_outlier"] = False
    config["multi_entities"] = None
    config["use_extended_qa"] = False
    include_outlier = current_accum_step >= total_accum_step - 1 - outlier_num  # 决定是否包含异常值

    # 如果达到时间，则决定是否包含异常值
    if include_outlier:
        config["include_outlier"] = True
        return config

    if current_accum_step % 3 == 0:  # 每3步一次，处理多重实体
        config["multi_entities"] = multi_entities
        return config

    if current_accum_step % 3 == 1:  # 如果是第1步，使用扩展QA
        config["use_extended_qa"] = use_extended_qa
        return config

    return config


def get_parameter_count(encoder):  # 获取模型参数数量
    param_count = 0.0
    for p in encoder.parameters():
        if p.requires_grad:  # 仅计算需要梯度的参数
            param_count += p.numel()  # 累加参数数量
    return param_count  # 返回参数总数


class Trainer:
    def __init__(self,
                 llm_model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,  # LLM模型
                 kbretriever: KBRetriever,  # 知识库检索类
                 tokenizer: transformers.PreTrainedTokenizer,  # 标记化器
                 kb_token_layer_frequency: int,  # 知识库トークンのレイヤーの頻度
                 num_steps: int,  # 总步数
                 lr: float,  # 学习率
                 device: torch.device | None,  # 设备
                 use_lr_decay: bool,  # 是否使用学习率衰减
                 kb_size: int | List[int],  # 知识库大小
                 llm_savename: str,  # LLM保存的名称
                 output_dir: str,  # 输出目录
                 sep_query_head: bool = False,  # 是否分离查询头
                 max_seq_len: int | None = None,  # 最大序列长度
                 ):
        self.accelerator = Accelerator()  # 创建加速器对象
        self.logger = logging.getLogger("training")  # 获取训练记录器
        self.tokenizer = tokenizer  # 保存tokenizer
        self.sep_query_head = sep_query_head  # 保存是否分离查询头
        self.kb_token_layer_frequency = kb_token_layer_frequency  # 保存知识库token层频率
        self.num_steps = num_steps  # 保存总步数
        self.lr = lr  # 保存学习率
        self.max_seq_len = max_seq_len  # 保存最大序列长度

        self.model = llm_model  # 保存LLM模型
        self.model.gradient_checkpointing_enable()  # 启用梯度检查点

        self.device = device if device is not None else self.accelerator.device  # 设置设备
        self.kbretriever = kbretriever  # 保存KBRetriever对象
        self.kb_size = kb_size  # 保存知识库大小
        self.use_lr_decay = use_lr_decay  # 保存是否使用学习率衰减
        self.llm_savename = llm_savename  # 保存LLM名称
        self.output_path = pathlib.Path(output_dir)  # 保存输出路径

        if isinstance(llm_model, KBLaMPhi3ForCausalLM):  # 如果是Phi3模型
            self._get_batch = partial(get_batch, format_QA_phi3, create_labels_for_phi3)  # 设置batch获取函数
            self._get_params = get_phi3_query_head_parameters  # 设置参数获取函数
        elif isinstance(llm_model, KblamLlamaForCausalLM):  # 如果是Llama模型
            self._get_batch = partial(get_batch, format_QA_llama, create_labels_for_llama)  # 设置batch获取函数
            self._get_params = get_llama3_query_head_parameters  # 设置参数获取函数
        else:
            raise ValueError(f"{llm_model} not recognised")  # 抛出错误，模型未识别

        self.scheduler, self.optim = self.setup_scheduler_and_optim()  # 设置调度器和优化器

        self.model, self.optim, self._get_batch, self.kbretriever.encoder = self.accelerator.prepare(self.model,
                                                                                                     self.optim,
                                                                                                     self._get_batch,
                                                                                                     self.kbretriever.encoder
                                                                                                     )

    def setup_scheduler_and_optim(self):  # 设置学习率调度器和优化器
        if self.sep_query_head:  # 如果分离查询头
            self.logger.info("Query head being fine tuned!")  # 记录信息：正在微调查询头
            llm_q_params = self._get_params(self.model, self.sep_query_head, self.kb_token_layer_frequency)  # 获取LLM查询参数
            scheduler, optim = setup_scheduler_and_optimizer(chain(self.kbretriever.encoder.parameters(), llm_q_params),
                                                             self.lr,
                                                             self.num_steps,
                                                             )
            self.logger.info("Optimizer recreated")  # 记录信息：优化器已重新创建
        else:
            scheduler, optim = setup_scheduler_and_optimizer(self.kbretriever.encoder.parameters(), self.lr,
                                                             self.num_steps
                                                             )
            self.logger.info("Optimizer recreated")  # 记录信息：优化器已重新创建
        return scheduler, optim  # 返回调度器和优化器

    def train(self,
              console,
              training_set: List[Dict],  # 训练集
              batch_size,  # 批处理大小
              grad_accum_steps: int,  # 梯度累积步数
              outlier_num: int,  # 异常值数量
              use_data_aug: bool = False,  # 是否使用数据增强
              multi_entities: bool = False,  # 是否使用多重实体
              use_extended_qa: bool = False,  # 是否使用扩展QA
              save_period: int = 2000,  # 保存周期
              resumed_step: int = 0,  # 恢复的步数
              kb_config: KBLaMConfig = None,  # KBLAM配置
              ):
        train_losses = []  # 保存训练损失
        start_step = resumed_step  # 开始的步数

        loss_fct = CrossEntropyLoss(reduction="none")  # 初始化交叉熵损失函数

        # 计算每个GPU的累积步数
        num_processes = self.accelerator.num_processes  # 获取进程数量
        accum_steps_per_gpu = max(1, grad_accum_steps // num_processes)  # 每个GPU的累积步数
        effective_batch_size = batch_size * grad_accum_steps  # 计算有效批处理大小

        if self.accelerator.is_main_process:  # 仅主进程记录
            self.logger.info(f"Training with {num_processes} GPUs")  # 记录信息：使用的GPU数量
            self.logger.info(
                f"Total accumulation steps: {grad_accum_steps}, Steps per GPU: {accum_steps_per_gpu}")  # 记录信息：总累积步骤和每个GPU步骤
            self.logger.info(f"Batch size: {batch_size}")  # 记录信息：批处理大小
            self.logger.info(f"Effective batch size: {effective_batch_size}")  # 记录信息：有效批处理大小

        with create_custom_progress_bar(console=console,
                                        disable=not self.accelerator.is_main_process) as pbar:  # 创建自定义进度条
            task = pbar.add_task("Training", total=self.num_steps, loss=100)  # 添加任务到进度条
            for step in range(start_step, self.num_steps, 1):  # 训练循环
                self.optim.zero_grad()  # 清零梯度
                losses = []  # 存储每个步骤的损失

                # 计算此GPU应处理的累积步骤
                process_rank = self.accelerator.process_index  # 获取当前进程的索引
                start_accum_step = process_rank * accum_steps_per_gpu  # 计算开始的累积步骤
                end_accum_step = min(start_accum_step + accum_steps_per_gpu, grad_accum_steps)  # 计算结束的累积步骤

                # 累积梯度
                for a_step in range(start_accum_step, end_accum_step):  # 遍历所有累积步骤
                    step_config = get_step_config(a_step,
                                                  grad_accum_steps,
                                                  use_data_aug,
                                                  outlier_num,
                                                  multi_entities,
                                                  use_extended_qa,
                                                  )  # 获取此步的配置
                    input_ids, attention_masks, labels, batch_indices = self._get_batch(training_set,  # 获取batch
                                                                                        self.tokenizer,
                                                                                        self.device,
                                                                                        B=batch_size,
                                                                                        random_sample=True,
                                                                                        **step_config,
                                                                                        )

                    if a_step == 0 and step % 10 == 0:  # 每10步记录一次输入ID形状
                        self.logger.info(f"INPUT IDs SHAPE: {input_ids.shape}")

                    if self.max_seq_len is not None:  # 如果最大序列长度不为空
                        input_ids = input_ids[:, : self.max_seq_len]  # 截取输入ID
                        attention_masks = attention_masks[:, : self.max_seq_len]  # 截取注意力掩码
                        labels = labels[:, : self.max_seq_len]  # 截取标签
                        if a_step == 0 and step % 10 == 0:  # 每10步记录一次截取后的输入ID形状
                            self.logger.info(f"TRUNCATED INPUT IDs SHAPE: {input_ids.shape}")

                    kb_embedding = self.kbretriever.get_key_embeddings(batch_indices, len(input_ids), step,
                                                                       self.kb_size)  # 获取知识库嵌入
                    out = self.model(input_ids=input_ids,  # 使用模型计算输出
                                     attention_mask=attention_masks,
                                     kb_kvs=kb_embedding,
                                     output_attentions=True,
                                     kb_config=kb_config,
                                     )
                    logits = out["logits"]  # 提取logits输出

                    # 显示真实标签和模型预测以快速检查模型
                    if a_step == 0 and step % 10 == 0:
                        batch_index = 0  # 选择批处理中的一个示例
                        max_logits = logits.argmax(axis=2)  # 获取模型的预测
                        decoded_pred = self.tokenizer.decode(max_logits[batch_index, :-1])  # 解码预测
                        sel_labels = labels[batch_index, :]  # 获取标签
                        sel_labels = sel_labels[sel_labels >= 0]  # 移除填充标记-100
                        decoded_gt = self.tokenizer.decode(sel_labels)  # 解码真实标签
                        self.logger.info(f"KB SHAPE: {kb_embedding[0].shape}")  # 记录知识库形状
                        self.logger.info(f"GT: {decoded_gt}")  # 记录真实标签
                        self.logger.info(f"PRED: {decoded_pred}")  # 记录预测
                        wandb.log({"kbsize": kb_embedding[0].shape})
                        shift_logits = logits[..., :-1, :].contiguous()  # 移位logits以对应标签
                        shift_labels = labels[..., 1:].contiguous()  # 移位标签
                        weights = (shift_labels > 0).sum(-1, keepdim=True).expand(-1, shift_labels.shape[
                            1]).contiguous()  # 计算权重
                        # 扁平化tokens
                        model_config = (self.model.config
                                        if not isinstance(self.model, DistributedDataParallel)
                                        else self.model.module.config
                                        )
                        shift_logits = shift_logits.view(-1, model_config.vocab_size)  # 重塑logits
                        shift_labels = shift_labels.view(-1)  # 重塑标签
                        weights = weights.view(-1)  # 重塑权重

                        shift_labels = shift_labels.to(shift_logits.device)  # 移动标签到logits设备

                        # 计算损失
                        loss = (loss_fct(shift_logits, shift_labels) * weights.max() / weights
                                ).mean()  # 确保每个样本的权重相等

                        self.accelerator.backward(loss)  # 反向传播
                        losses.append(loss.item())  # 保存当前损失

                    self.optim.step()  # 优化器更新
                    if self.use_lr_decay:
                        self.scheduler.step()  # 更新学习率

                    # 收集并平均所有GPU的损失以进行报告
                    if losses:  # 仅当该GPU处理了任何批次时
                        local_loss = torch.tensor(np.mean(losses), device=self.device)  # 计算本地损失
                    else:
                        local_loss = torch.tensor(0.0, device=self.device)  # 如果没有损失，则设为0

                    # 收集所有进程的损失
                    all_losses = self.accelerator.gather(local_loss)  # 聚合损失
                    valid_losses = all_losses[all_losses > 0]  # 过滤掉未处理批次的零
                    avg_loss = valid_losses.mean().item() if len(valid_losses) > 0 else 0.0  # 计算平均损失

                    # 仅从主进程记录
                    if self.accelerator.is_main_process:
                        self.logger.info(f"step: {step}, loss: {avg_loss}")  # 记录训练步数和损失
                        wandb.log({'train_loss': np.mean(losses)})  # 记录到wandb
                        train_losses.append(avg_loss)  # 保存训练损失
                        pbar.update(task, advance=1, loss=avg_loss)  # 更新进度条

                    # 每save_period步保存一次模型
                    if (step % save_period) == 0 and (step != start_step):
                        try:
                            # 在同步之前记录内存使用情况
                            self.logger.info(
                                f"Is main process: {self.accelerator.is_main_process}, GPU memory before save: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB"
                            )

                            # 尝试释放内存
                            torch.cuda.empty_cache()  # 清除缓存

                            # 在保存之前进行同步
                            self.accelerator.wait_for_everyone()

                            if self.accelerator.is_main_process:
                                self.logger.info("Saving checkpoint...")  # 记录保存检查点的信息
                                self.logger.info("Making dirs...")  # 记录正在创建目录
                                # 保存模型 - 使用适当的目录创建
                                model_ckpt_name = self.output_path / f"{self.llm_savename}_step_{step}"
                                model_ckpt_name.mkdir(parents=True, exist_ok=True)  # 创建模型检查点目录

                                # 还创建编码器目录
                                encoder_dir = self.output_path / f"{self.llm_savename}_step_{step}_encoder"
                                encoder_dir.mkdir(parents=True, exist_ok=True)  # 创建编码器保存目录

                                self.logger.info("Saving model...")  # 记录保存模型的信息
                                # 解包并保存模型
                                unwrapped_model = self.accelerator.unwrap_model(self.model)
                                unwrapped_model.save_pretrained(model_ckpt_name,
                                                                is_main_process=self.accelerator.is_main_process,
                                                                save_function=self.accelerator.save,
                                                                )

                                self.logger.info("Saving encoder...")  # 记录保存编码器的信息
                                # 从主进程保存编码器和配置
                                encoder_ckpt_name = encoder_dir / "encoder.pt"
                                torch.save(self.kbretriever.encoder.state_dict(), encoder_ckpt_name)  # 保存编码器状态

                                self.logger.info("Saving config...")  # 记录保存配置的信息
                                # 明确保存配置为JSON
                                config_path = model_ckpt_name / "kb_config_explicit.json"
                                with open(config_path, 'w') as f:
                                    f.write(kb_config.to_json_string())  # 保存KB配置为JSON字符串

                        except Exception as e:
                            self.logger.error(f"Error saving checkpoint: {e}")  # 记录保存检查点时的错误
                            self.logger.error(f"Error details: {str(e)}")  # 记录错误详细信息
                            raise e  # 抛出异常


if __name__ == "__main__":
    args = get_arguments()
    dataset_name = args.train_dataset  # 获取训练数据集名称
    seed = args.seed  # 获取随机种子
    num_train_sample = args.num_train_sample  # 获取数据集大小
    batch_size = args.batch_size  # 获取批处理大小
    total_step = args.total_steps  # 获取总步数
    encoder_name = args.encoder_name  # 获取编码器规格
    key_embed_source = args.key_embed_source  # 获取键嵌入来源
    use_data_augment = args.use_data_augment  # 是否使用数据增强
    use_learning_rate_decay = args.use_learning_rate_decay  # 是否使用学习率衰减
    use_cached_embed = args.use_cached_embed  # 是否使用缓存的嵌入
    dataset_dir = args.dataset_dir  # 数据集目录
    model_dir_to_resume = args.model_dir_to_resume  # 继续训练的模型目录
    model_save_dir = args.model_save_dir  # 模型保存目录
    separate_query_head = args.separate_query_head  # 是否分离查询头
    kb_size = args.kb_size  # 知识库大小
    dynamic_kb_size = args.dynamic_kb_size  # 动态知识库大小
    max_seq_len = args.max_seq_len  # 最大序列长度
    gradient_accumulation_step = args.gradient_accumulation_step  # 获取梯度累积步数
    length_invariance = args.length_invariance  # 是否长短不变
    outlier_num = args.outlier_num  # 异常值数量
    multi_entity = args.multi_entity  # 多重实体数量
    use_extended_qa = args.use_extended_qa  # 是否使用扩展QA
    kb_token_layer_frequency = args.kb_token_layer_frequency  # KB token层频率
    llm_type = args.llm_type  # LLM类型
    hf_model_specification = args.hf_model_specification  # HuggingFace模型规格
    hf_token = args.hf_token  # HuggingFace令牌
    print(vars(args))  # 打印参数字典

    torch.manual_seed(seed)  # 设置PyTorch随机种子
    np.random.seed(seed)  # 设置NumPy随机种子

    # 设置NCCL超时
    os.environ["NCCL_TIMEOUT"] = "1200000"
    if torch.cuda.is_available(): device = torch.device("cuda")  # 使用CUDA设备

    if kb_size is not None and dynamic_kb_size is not None:
        raise ValueError("Can't specify kb_size and dynamic_kb_size. Use only one")  # 不能同时指定两个参数

    if kb_size is not None:
        kb_size = kb_size
    else:
        kb_size = dynamic_kb_size

    pathlib.Path(model_save_dir).mkdir(parents=True, exist_ok=True)  # 创建模型保存目录

    # 尝试释放内存
    torch.cuda.empty_cache()

    if use_cached_embed:  # 如果使用缓存的嵌入
        # 我们从磁盘加载预计算版本，快速化处理
        key_embeds, value_embeds = load_cached_embeddings(encoder_specification=encoder_name,
                                                          dataset_dir=dataset_dir,
                                                          dataset_name=dataset_name,
                                                          key_embed_source=key_embed_source)  # 加载缓存的嵌入

    if use_extended_qa:  # 如果使用扩展QA
        dataset = json.load(open(os.path.join(dataset_dir, f"{dataset_name}_augmented.json")))  # 加载扩展数据集
    else:
        # dataset = json.load(open(os.path.join(dataset_dir, f"{dataset_name}.json")))  # 加载常规数据集
        dataset = json.load(open(os.path.join(f"enron.json")))

    training_set = dataset[:num_train_sample]  # 划分出训练集

    # 设置LLM模型
    # 选择恢复模型或HuggingFace模型规格
    if model_dir_to_resume:
        llm_model_specification = model_dir_to_resume
    else:
        llm_model_specification = hf_model_specification

    # 获取恢复的步骤
    if not model_dir_to_resume:
        resumed_step = 0
    else:
        resumed_step = int(model_dir_to_resume.split("_")[-1])

    # 必须提供模型目录或HuggingFace模型规格
    if llm_model_specification is None:
        raise ValueError("Either supply model_dir_to_resume or hf_model_spec")

    hf_token = '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'
    if hf_token is None and args.llm_type == "llama3":  # 如果使用Llama3模型且未提供令牌
        raise ValueError("Please supply HuggingFace token when loading model Llama weights from HuggingFace")

    # Tokenizer来自基础模型
    if hf_token is not None and args.llm_type == "llama3":
        hf_token = hf_token
    else:
        hf_token = None

    print(f"AutoTokenizer.from_pretrained(pretrained_model_name_or_path={hf_model_specification}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=hf_model_specification,
                                              trust_remote_code=True,  # 'meta-llama/Llama-3.2-1B-Instruct'
                                              token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    if args.llm_type == "llama3":
        model = KblamLlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=llm_model_specification,
                                                      device_map=device,
                                                      torch_dtype=torch.bfloat16,
                                                      trust_remote_code=True,
                                                      token=hf_token)
    elif args.llm_type == "phi3":  # 如果选择Phi3
        model = KBLaMPhi3ForCausalLM.from_pretrained(llm_model_specification,
                                                     device_map=device,
                                                     torch_dtype="auto",
                                                     trust_remote_code=True)
    else:
        assert False, f"LLM type {args.llm_type} not recognised"

    model.eval()
    for _, param in model.named_parameters(): param.requires_grad = False

    # 设置编码器
    out_dim = model.config.hidden_size * (model.config.num_hidden_layers // kb_token_layer_frequency + 1)
    encoder = KBEncoder(encoder_name=encoder_name,  # 创建知识库编码器
                        projector_type="linear",
                        endpoint_url="",
                        out_dim=out_dim,
                        frozen_base_model=True,  # 冻结基础模型
                        device=device)

    if model_dir_to_resume:  # 如果提供了恢复目录
        encoder.load_state_dict(torch.load(os.path.join(model_dir_to_resume, "encoder.pt")))  # 加载编码器状态
        kb_config = KBLaMConfig.from_pretrained(os.path.join(model_dir_to_resume, "kb_config.json"))  # 加载KB配置
    else:
        kb_config = KBLaMConfig(separate_query_head=separate_query_head,
                                kb_token_layer_frequency=kb_token_layer_frequency)

    encoder.train()  # 设置编码器为训练模式

    kbretriever = KBRetriever(encoder=encoder,  # 创建KBRetriever对象
                              dataset=training_set,
                              key_embds=key_embeds,  # 用加载的键嵌入
                              value_embds=value_embeds)

    # 创建checkpoint名称
    prefix_string = get_prefix_str(args=args)  # 获取实验前缀字符串
    llm_ckpt_name = f"{prefix_string}KeyFrom{key_embed_source}_{encoder_name}_{dataset_name}_{llm_type}"

    # 开始训练
    trainer = Trainer(llm_model=model,  # 创建Trainer对象
                      kbretriever=kbretriever,
                      tokenizer=tokenizer,
                      kb_token_layer_frequency=kb_token_layer_frequency,
                      num_steps=total_step,
                      lr=args.lr,
                      device=device,
                      use_lr_decay=use_learning_rate_decay,
                      kb_size=kb_size,  # 传递知识库大小
                      llm_savename=llm_ckpt_name,
                      output_dir=model_save_dir,
                      sep_query_head=separate_query_head,
                      max_seq_len=max_seq_len)

    trainer.train(training_set=training_set,  # 开始训练过程
                  batch_size=batch_size,
                  grad_accum_steps=gradient_accumulation_step,
                  outlier_num=outlier_num,
                  use_data_aug=use_data_augment,
                  multi_entities=multi_entity,
                  use_extended_qa=use_extended_qa,
                  save_period=3000,  # 设置保存周期
                  resumed_step=resumed_step,  # 设置恢复步数
                  kb_config=kb_config)
