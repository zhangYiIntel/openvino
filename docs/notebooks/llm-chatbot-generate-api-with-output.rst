Create an LLM-powered Chatbot using OpenVINO Generate API
=========================================================

In the rapidly evolving world of artificial intelligence (AI), chatbots
have emerged as powerful tools for businesses to enhance customer
interactions and streamline operations. Large Language Models (LLMs) are
artificial intelligence systems that can understand and generate human
language. They use deep learning algorithms and massive amounts of data
to learn the nuances of language and produce coherent and relevant
responses. While a decent intent-based chatbot can answer basic,
one-touch inquiries like order management, FAQs, and policy questions,
LLM chatbots can tackle more complex, multi-touch questions. LLM enables
chatbots to provide support in a conversational manner, similar to how
humans do, through contextual memory. Leveraging the capabilities of
Language Models, chatbots are becoming increasingly intelligent, capable
of understanding and responding to human language with remarkable
accuracy.

Previously, we already discussed how to build an instruction-following
pipeline using OpenVINO and Optimum Intel, please check out `Dolly
example <dolly-2-instruction-following-with-output.html>`__ for reference. In this
tutorial, we consider how to use the power of OpenVINO for running Large
Language Models for chat. We will use a pre-trained model from the
`Hugging Face
Transformers <https://huggingface.co/docs/transformers/index>`__
library. The `Hugging Face Optimum
Intel <https://huggingface.co/docs/optimum/intel/index>`__ library
converts the models to OpenVINO™ IR format. To simplify the user
experience, we will use `OpenVINO Generate
API <https://github.com/openvinotoolkit/openvino.genai>`__ for
generation of instruction-following inference pipeline.

The tutorial consists of the following steps:

-  Install prerequisites
-  Download and convert the model from a public source using the
   `OpenVINO integration with Hugging Face
   Optimum <https://huggingface.co/blog/openvino>`__.
-  Compress model weights to 4-bit or 8-bit data types using
   `NNCF <https://github.com/openvinotoolkit/nncf>`__
-  Create a chat inference pipeline with `OpenVINO Generate
   API <https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md>`__.
-  Run chat pipeline

**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Select model for inference <#select-model-for-inference>`__
-  `Convert model using Optimum-CLI
   tool <#convert-model-using-optimum-cli-tool>`__
-  `Compress model weights <#compress-model-weights>`__

   -  `Weights Compression using
      Optimum-CLI <#weights-compression-using-optimum-cli>`__
   -  `Weight compression with AWQ <#weight-compression-with-awq>`__

      -  `Select device for inference and model
         variant <#select-device-for-inference-and-model-variant>`__

-  `Instantiate pipeline with OpenVINO Generate
   API <#instantiate-pipeline-with-openvino-generate-api>`__
-  `Run Chatbot <#run-chatbot>`__

   -  `Prepare text streamer to get results
      runtime <#prepare-text-streamer-to-get-results-runtime>`__
   -  `Setup of the chatbot life process
      function <#setup-of-the-chatbot-life-process-function>`__
   -  `Next Step <#next-step>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



Install required dependencies

.. code:: ipython3

    import os

    os.environ["GIT_CLONE_PROTECTION_ACTIVE"] = "false"

    %pip install -Uq pip
    %pip uninstall -q -y optimum optimum-intel
    %pip install -q -U "openvino>=2024.3.0" openvino-tokenizers[transformers] openvino-genai
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu\
    "git+https://github.com/huggingface/optimum-intel.git"\
    "git+https://github.com/openvinotoolkit/nncf.git"\
    "torch>=2.1"\
    "datasets" \
    "accelerate" \
    "gradio>=4.19" \
    "onnx<=1.16.1; sys_platform=='win32'" "einops" "transformers_stream_generator" "tiktoken" "bitsandbytes"

.. code:: ipython3

    import os
    from pathlib import Path
    import requests
    import shutil

    # fetch model configuration

    config_shared_path = Path("../../utils/llm_config.py")
    config_dst_path = Path("llm_config.py")

    if not config_dst_path.exists():
        if config_shared_path.exists():
            try:
                os.symlink(config_shared_path, config_dst_path)
            except Exception:
                shutil.copy(config_shared_path, config_dst_path)
        else:
            r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/llm_config.py")
            with open("llm_config.py", "w", encoding="utf-8") as f:
                f.write(r.text)
    elif not os.path.islink(config_dst_path):
        print("LLM config will be updated")
        if config_shared_path.exists():
            shutil.copy(config_shared_path, config_dst_path)
        else:
            r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/llm_config.py")
            with open("llm_config.py", "w", encoding="utf-8") as f:
                f.write(r.text)

Select model for inference
--------------------------



The tutorial supports different models, you can select one from the
provided options to compare the quality of open source LLM solutions.
>\ **Note**: conversion of some models can require additional actions
from user side and at least 64GB RAM for conversion.

.. raw:: html

   <details>

.. raw:: html

   <summary>

Click here to see available models options

.. raw:: html

   </summary>

-  **tiny-llama-1b-chat** - This is the chat model finetuned on top of
   `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T <https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T>`__.
   The TinyLlama project aims to pretrain a 1.1B Llama model on 3
   trillion tokens with the adoption of the same architecture and
   tokenizer as Llama 2. This means TinyLlama can be plugged and played
   in many open-source projects built upon Llama. Besides, TinyLlama is
   compact with only 1.1B parameters. This compactness allows it to
   cater to a multitude of applications demanding a restricted
   computation and memory footprint. More details about model can be
   found in `model
   card <https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0>`__
-  **mini-cpm-2b-dpo** - MiniCPM is an End-Size LLM developed by
   ModelBest Inc. and TsinghuaNLP, with only 2.4B parameters excluding
   embeddings. After Direct Preference Optimization (DPO) fine-tuning,
   MiniCPM outperforms many popular 7b, 13b and 70b models. More details
   can be found in
   `model_card <https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp16>`__.
-  **gemma-2b-it** - Gemma is a family of lightweight, state-of-the-art
   open models from Google, built from the same research and technology
   used to create the Gemini models. They are text-to-text, decoder-only
   large language models, available in English, with open weights,
   pre-trained variants, and instruction-tuned variants. Gemma models
   are well-suited for a variety of text generation tasks, including
   question answering, summarization, and reasoning. This model is
   instruction-tuned version of 2B parameters model. More details about
   model can be found in `model
   card <https://huggingface.co/google/gemma-2b-it>`__. >\ **Note**: run
   model with demo, you will need to accept license agreement. >You must
   be a registered user in Hugging Face Hub. Please visit
   `HuggingFace model
   card <https://huggingface.co/google/gemma-2b-it>`__, carefully read
   terms of usage and click accept button. You will need to use an
   access token for the code below to run. For more information on
   access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   >You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: python

       ## login to huggingfacehub to get access to pretrained model



       from huggingface_hub import notebook_login, whoami

       try:
           whoami()
           print('Authorization token already provided')
       except OSError:
           notebook_login()

-  **phi3-mini-instruct** - The Phi-3-Mini is a 3.8B parameters,
   lightweight, state-of-the-art open model trained with the Phi-3
   datasets that includes both synthetic data and the filtered publicly
   available websites data with a focus on high-quality and reasoning
   dense properties. More details about model can be found in `model
   card <https://huggingface.co/microsoft/Phi-3-mini-4k-instruct>`__,
   `Microsoft blog <https://aka.ms/phi3blog-april>`__ and `technical
   report <https://aka.ms/phi3-tech-report>`__.
-  **red-pajama-3b-chat** - A 2.8B parameter pre-trained language model
   based on GPT-NEOX architecture. It was developed by Together Computer
   and leaders from the open-source AI community. The model is
   fine-tuned on OASST1 and Dolly2 datasets to enhance chatting ability.
   More details about model can be found in `HuggingFace model
   card <https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1>`__.
-  **gemma-7b-it** - Gemma is a family of lightweight, state-of-the-art
   open models from Google, built from the same research and technology
   used to create the Gemini models. They are text-to-text, decoder-only
   large language models, available in English, with open weights,
   pre-trained variants, and instruction-tuned variants. Gemma models
   are well-suited for a variety of text generation tasks, including
   question answering, summarization, and reasoning. This model is
   instruction-tuned version of 7B parameters model. More details about
   model can be found in `model
   card <https://huggingface.co/google/gemma-7b-it>`__. >\ **Note**: run
   model with demo, you will need to accept license agreement. >You must
   be a registered user in Hugging Face Hub. Please visit
   `HuggingFace model
   card <https://huggingface.co/google/gemma-7b-it>`__, carefully read
   terms of usage and click accept button. You will need to use an
   access token for the code below to run. For more information on
   access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   >You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: python

       ## login to huggingfacehub to get access to pretrained model

       from huggingface_hub import notebook_login, whoami

       try:
           whoami()
           print('Authorization token already provided')
       except OSError:
           notebook_login()

-  **llama-2-7b-chat** - LLama 2 is the second generation of LLama
   models developed by Meta. Llama 2 is a collection of pre-trained and
   fine-tuned generative text models ranging in scale from 7 billion to
   70 billion parameters. llama-2-7b-chat is 7 billions parameters
   version of LLama 2 finetuned and optimized for dialogue use case.
   More details about model can be found in the
   `paper <https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/>`__,
   `repository <https://github.com/facebookresearch/llama>`__ and
   `HuggingFace model
   card <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`__.
   >\ **Note**: run model with demo, you will need to accept license
   agreement. >You must be a registered user in Hugging Face Hub.
   Please visit `HuggingFace model
   card <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`__,
   carefully read terms of usage and click accept button. You will need
   to use an access token for the code below to run. For more
   information on access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   >You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: python

       ## login to huggingfacehub to get access to pretrained model

       from huggingface_hub import notebook_login, whoami

       try:
           whoami()
           print('Authorization token already provided')
       except OSError:
           notebook_login()

-  **llama-3-8b-instruct** - Llama 3 is an auto-regressive language
   model that uses an optimized transformer architecture. The tuned
   versions use supervised fine-tuning (SFT) and reinforcement learning
   with human feedback (RLHF) to align with human preferences for
   helpfulness and safety. The Llama 3 instruction tuned models are
   optimized for dialogue use cases and outperform many of the available
   open source chat models on common industry benchmarks. More details
   about model can be found in `Meta blog
   post <https://ai.meta.com/blog/meta-llama-3/>`__, `model
   website <https://llama.meta.com/llama3>`__ and `model
   card <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`__.
   >\ **Note**: run model with demo, you will need to accept license
   agreement. >You must be a registered user in Hugging Face Hub.
   Please visit `HuggingFace model
   card <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`__,
   carefully read terms of usage and click accept button. You will need
   to use an access token for the code below to run. For more
   information on access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   >You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: python

       ## login to huggingfacehub to get access to pretrained model

       from huggingface_hub import notebook_login, whoami

       try:
           whoami()
           print('Authorization token already provided')
       except OSError:
           notebook_login()

-  **llama-3.1-8b-instruct** - The Llama 3.1 instruction tuned text only
   models (8B, 70B, 405B) are optimized for multilingual dialogue use
   cases and outperform many of the available open source and closed
   chat models on common industry benchmarks. More details about model
   can be found in `Meta blog
   post <https://ai.meta.com/blog/meta-llama-3-1/>`__, `model
   website <https://llama.meta.com>`__ and `model
   card <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct>`__.
   >\ **Note**: run model with demo, you will need to accept license
   agreement. >You must be a registered user in Hugging Face Hub.
   Please visit `HuggingFace model
   card <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct>`__,
   carefully read terms of usage and click accept button. You will need
   to use an access token for the code below to run. For more
   information on access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   >You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: python

       ## login to huggingfacehub to get access to pretrained model

       from huggingface_hub import notebook_login, whoami

       try:
           whoami()
           print('Authorization token already provided')
       except OSError:
           notebook_login()

-  **qwen2-1.5b-instruct/qwen2-7b-instruct** - Qwen2 is the new series
   of Qwen large language models.Compared with the state-of-the-art open
   source language models, including the previous released Qwen1.5,
   Qwen2 has generally surpassed most open source models and
   demonstrated competitiveness against proprietary models across a
   series of benchmarks targeting for language understanding, language
   generation, multilingual capability, coding, mathematics, reasoning,
   etc. For more details, please refer to
   `model_card <https://huggingface.co/Qwen/Qwen2-7B-Instruct>`__,
   `blog <https://qwenlm.github.io/blog/qwen2/>`__,
   `GitHub <https://github.com/QwenLM/Qwen2>`__, and
   `Documentation <https://qwen.readthedocs.io/en/latest/>`__.
-  **qwen1.5-0.5b-chat/qwen1.5-1.8b-chat/qwen1.5-7b-chat** - Qwen1.5 is
   the beta version of Qwen2, a transformer-based decoder-only language
   model pretrained on a large amount of data. Qwen1.5 is a language
   model series including decoder language models of different model
   sizes. It is based on the Transformer architecture with SwiGLU
   activation, attention QKV bias, group query attention, mixture of
   sliding window attention and full attention. You can find more
   details about model in the `model
   repository <https://huggingface.co/Qwen>`__.
-  **qwen-7b-chat** - Qwen-7B is the 7B-parameter version of the large
   language model series, Qwen (abbr. Tongyi Qianwen), proposed by
   Alibaba Cloud. Qwen-7B is a Transformer-based large language model,
   which is pretrained on a large volume of data, including web texts,
   books, codes, etc. For more details about Qwen, please refer to the
   `GitHub <https://github.com/QwenLM/Qwen>`__ code repository.
-  **chatglm3-6b** - ChatGLM3-6B is the latest open-source model in the
   ChatGLM series. While retaining many excellent features such as
   smooth dialogue and low deployment threshold from the previous two
   generations, ChatGLM3-6B employs a more diverse training dataset,
   more sufficient training steps, and a more reasonable training
   strategy. ChatGLM3-6B adopts a newly designed `Prompt
   format <https://github.com/THUDM/ChatGLM3/blob/main/PROMPT_en.md>`__,
   in addition to the normal multi-turn dialogue. You can find more
   details about model in the `model
   card <https://huggingface.co/THUDM/chatglm3-6b>`__
-  **mistral-7b** - The Mistral-7B-v0.1 Large Language Model (LLM) is a
   pretrained generative text model with 7 billion parameters. You can
   find more details about model in the `model
   card <https://huggingface.co/mistralai/Mistral-7B-v0.1>`__,
   `paper <https://arxiv.org/abs/2310.06825>`__ and `release blog
   post <https://mistral.ai/news/announcing-mistral-7b/>`__.
-  **zephyr-7b-beta** - Zephyr is a series of language models that are
   trained to act as helpful assistants. Zephyr-7B-beta is the second
   model in the series, and is a fine-tuned version of
   `mistralai/Mistral-7B-v0.1 <https://huggingface.co/mistralai/Mistral-7B-v0.1>`__
   that was trained on on a mix of publicly available, synthetic
   datasets using `Direct Preference Optimization
   (DPO) <https://arxiv.org/abs/2305.18290>`__. You can find more
   details about model in `technical
   report <https://arxiv.org/abs/2310.16944>`__ and `HuggingFace model
   card <https://huggingface.co/HuggingFaceH4/zephyr-7b-beta>`__.
-  **neural-chat-7b-v3-1** - Mistral-7b model fine-tuned using Intel
   Gaudi. The model fine-tuned on the open source dataset
   `Open-Orca/SlimOrca <https://huggingface.co/datasets/Open-Orca/SlimOrca>`__
   and aligned with `Direct Preference Optimization (DPO)
   algorithm <https://arxiv.org/abs/2305.18290>`__. More details can be
   found in `model
   card <https://huggingface.co/Intel/neural-chat-7b-v3-1>`__ and `blog
   post <https://medium.com/@NeuralCompressor/the-practice-of-supervised-finetuning-and-direct-preference-optimization-on-habana-gaudi2-a1197d8a3cd3>`__.
-  **notus-7b-v1** - Notus is a collection of fine-tuned models using
   `Direct Preference Optimization
   (DPO) <https://arxiv.org/abs/2305.18290>`__. and related
   `RLHF <https://huggingface.co/blog/rlhf>`__ techniques. This model is
   the first version, fine-tuned with DPO over zephyr-7b-sft. Following
   a data-first approach, the only difference between Notus-7B-v1 and
   Zephyr-7B-beta is the preference dataset used for dDPO. Proposed
   approach for dataset creation helps to effectively fine-tune Notus-7b
   that surpasses Zephyr-7B-beta and Claude 2 on
   `AlpacaEval <https://tatsu-lab.github.io/alpaca_eval/>`__. More
   details about model can be found in `model
   card <https://huggingface.co/argilla/notus-7b-v1>`__.
-  **youri-7b-chat** - Youri-7b-chat is a Llama2 based model. `Rinna
   Co., Ltd. <https://rinna.co.jp/>`__ conducted further pre-training
   for the Llama2 model with a mixture of English and Japanese datasets
   to improve Japanese task capability. The model is publicly released
   on Hugging Face hub. You can find detailed information at the
   `rinna/youri-7b-chat project
   page <https://huggingface.co/rinna/youri-7b>`__.
-  **baichuan2-7b-chat** - Baichuan 2 is the new generation of
   large-scale open-source language models launched by `Baichuan
   Intelligence inc <https://www.baichuan-ai.com/home>`__. It is trained
   on a high-quality corpus with 2.6 trillion tokens and has achieved
   the best performance in authoritative Chinese and English benchmarks
   of the same size.
-  **internlm2-chat-1.8b** - InternLM2 is the second generation InternLM
   series. Compared to the previous generation model, it shows
   significant improvements in various capabilities, including
   reasoning, mathematics, and coding. More details about model can be
   found in `model repository <https://huggingface.co/internlm>`__.
-  **glm-4-9b-chat** - GLM-4-9B is the open-source version of the latest
   generation of pre-trained models in the GLM-4 series launched by
   Zhipu AI. In the evaluation of data sets in semantics, mathematics,
   reasoning, code, and knowledge, GLM-4-9B and its human
   preference-aligned version GLM-4-9B-Chat have shown superior
   performance beyond Llama-3-8B. In addition to multi-round
   conversations, GLM-4-9B-Chat also has advanced features such as web
   browsing, code execution, custom tool calls (Function Call), and long
   text reasoning (supporting up to 128K context). More details about
   model can be found in `model
   card <https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/README_en.md>`__,
   `technical report <https://arxiv.org/pdf/2406.12793>`__ and
   `repository <https://github.com/THUDM/GLM-4>`__

.. raw:: html

   </details>

.. code:: ipython3

    from llm_config import SUPPORTED_LLM_MODELS
    import ipywidgets as widgets

.. code:: ipython3

    model_languages = list(SUPPORTED_LLM_MODELS)

    model_language = widgets.Dropdown(
        options=model_languages,
        value=model_languages[0],
        description="Model Language:",
        disabled=False,
    )

    model_language




.. parsed-literal::

    Dropdown(description='Model Language:', options=('English', 'Chinese', 'Japanese'), value='English')



.. code:: ipython3

    model_ids = list(SUPPORTED_LLM_MODELS[model_language.value])

    model_id = widgets.Dropdown(
        options=model_ids,
        value=model_ids[0],
        description="Model:",
        disabled=False,
    )

    model_id




.. parsed-literal::

    Dropdown(description='Model:', options=('qwen2-0.5b-instruct', 'tiny-llama-1b-chat', 'qwen2-1.5b-instruct', 'g…



.. code:: ipython3

    model_configuration = SUPPORTED_LLM_MODELS[model_language.value][model_id.value]
    print(f"Selected model {model_id.value}")


.. parsed-literal::

    Selected model llama-3.1-8b-instruct


Convert model using Optimum-CLI tool
------------------------------------



`Optimum Intel <https://huggingface.co/docs/optimum/intel/index>`__
is the interface between the
`Transformers <https://huggingface.co/docs/transformers/index>`__ and
`Diffusers <https://huggingface.co/docs/diffusers/index>`__ libraries
and OpenVINO to accelerate end-to-end pipelines on Intel architectures.
It provides ease-to-use cli interface for exporting models to `OpenVINO
Intermediate Representation
(IR) <https://docs.openvino.ai/2024/documentation/openvino-ir-format.html>`__
format.

The command bellow demonstrates basic command for model export with
``optimum-cli``

::

   optimum-cli export openvino --model <model_id_or_path> --task <task> <out_dir>

where ``--model`` argument is model id from HuggingFace Hub or local
directory with model (saved using ``.save_pretrained`` method),
``--task`` is one of `supported
task <https://huggingface.co/docs/optimum/exporters/task_manager>`__
that exported model should solve. For LLMs it will be
``text-generation-with-past``. If model initialization requires to use
remote code, ``--trust-remote-code`` flag additionally should be passed.

Compress model weights
----------------------



The `Weights
Compression <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html>`__
algorithm is aimed at compressing the weights of the models and can be
used to optimize the model footprint and performance of large models
where the size of weights is relatively larger than the size of
activations, for example, Large Language Models (LLM). Compared to INT8
compression, INT4 compression improves performance even more, but
introduces a minor drop in prediction quality.

Weights Compression using Optimum-CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



You can also apply fp16, 8-bit or 4-bit weight compression on the
Linear, Convolutional and Embedding layers when exporting your model
with the CLI by setting ``--weight-format`` to respectively fp16, int8
or int4. This type of optimization allows to reduce the memory footprint
and inference latency. By default the quantization scheme for int8/int4
will be
`asymmetric <https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md#asymmetric-quantization>`__,
to make it
`symmetric <https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md#symmetric-quantization>`__
you can add ``--sym``.

For INT4 quantization you can also specify the following arguments :

- The ``--group-size`` parameter will define the group size to use for
  quantization, -1 it will results in per-column quantization.
- The ``--ratio`` parameter controls the ratio between 4-bit and 8-bit
  quantization. If set to 0.9, it means that 90% of the layers will be
  quantized to int4 while 10% will be quantized to int8.

Smaller group_size and ratio values usually improve accuracy at the
sacrifice of the model size and inference latency.

   **Note**: There may be no speedup for INT4/INT8 compressed models on
   dGPU.

.. code:: ipython3

    from IPython.display import display

    prepare_int4_model = widgets.Checkbox(
        value=True,
        description="Prepare INT4 model",
        disabled=False,
    )
    prepare_int8_model = widgets.Checkbox(
        value=False,
        description="Prepare INT8 model",
        disabled=False,
    )
    prepare_fp16_model = widgets.Checkbox(
        value=False,
        description="Prepare FP16 model",
        disabled=False,
    )

    display(prepare_int4_model)
    display(prepare_int8_model)
    display(prepare_fp16_model)



.. parsed-literal::

    Checkbox(value=True, description='Prepare INT4 model')



.. parsed-literal::

    Checkbox(value=False, description='Prepare INT8 model')



.. parsed-literal::

    Checkbox(value=False, description='Prepare FP16 model')


Weight compression with AWQ
~~~~~~~~~~~~~~~~~~~~~~~~~~~



`Activation-aware Weight
Quantization <https://arxiv.org/abs/2306.00978>`__ (AWQ) is an algorithm
that tunes model weights for more accurate INT4 compression. It slightly
improves generation quality of compressed LLMs, but requires significant
additional time for tuning weights on a calibration dataset. We use
``wikitext-2-raw-v1/train`` subset of the
`Wikitext <https://huggingface.co/datasets/Salesforce/wikitext>`__
dataset for calibration.

Below you can enable AWQ to be additionally applied during model export
with INT4 precision.

   **Note**: Applying AWQ requires significant memory and time.

..

   **Note**: It is possible that there will be no matching patterns in
   the model to apply AWQ, in such case it will be skipped.

.. code:: ipython3

    enable_awq = widgets.Checkbox(
        value=False,
        description="Enable AWQ",
        disabled=not prepare_int4_model.value,
    )
    display(enable_awq)



.. parsed-literal::

    Checkbox(value=False, description='Enable AWQ')


We can now save floating point and compressed model variants

.. code:: ipython3

    from pathlib import Path
    from llm_config import compression_configs, get_optimum_cli_command
    from IPython.display import Markdown, display

    pt_model_id = model_configuration["model_id"]
    pt_model_name = model_id.value.split("-")[0]
    fp16_model_dir = Path(model_id.value) / "FP16"
    int8_model_dir = Path(model_id.value) / "INT8_compressed_weights"
    int4_model_dir = Path(model_id.value) / "INT4_compressed_weights"
    remote_code = model_configuration.get("remote_code", False)

    if prepare_fp16_model.value:
        if (fp16_model_dir / "openvino_model.xml").exists():
            print(f"✅ FP16 {model_id.value} model already converted and can be found in {fp16_model_dir}")
        else:
            print(f"⌛ {model_id.value} conversion to FP16 started. It may takes some time.")
            export_command = get_optimum_cli_command(pt_model_id, "fp16", fp16_model_dir, trust_remote_code=remote_code)
            display(Markdown("**Export command:**"))
            display(Markdown(f"`{export_command}`"))
            ! $export_command
            print(f"✅ FP16 {model_id.value} model converted and can be found in {fp16_model_dir}")

    if prepare_int8_model.value:
        if (int8_model_dir / "openvino_model.xml").exists():
            print(f"✅ INT8 {model_id.value} model already converted and can be found in {int8_model_dir}")
        else:
            print(f"⌛ {model_id.value} conversion to INT8 started. It may takes some time.")
            export_command = get_optimum_cli_command(pt_model_id, "int8", int8_model_dir, trust_remote_code=remote_code)
            display(Markdown("**Export command:**"))
            display(Markdown(f"`{export_command}`"))
            ! $export_command
            print(f"✅ INT8 {model_id.value} model converted and can be found in {int8_model_dir}")

    if prepare_int4_model.value:
        if (int4_model_dir / "openvino_model.xml").exists():
            print(f"✅ INT4 {model_id.value} model already converted and can be found in {int8_model_dir}")
        else:
            print(f"⌛ {model_id.value} conversion to INT4 started. It may takes some time.")
            model_compression_params = compression_configs.get(model_id.value, compression_configs["default"])
            export_command = get_optimum_cli_command(pt_model_id, "int4", int4_model_dir, model_compression_params, enable_awq.value, remote_code)
            display(Markdown("**Export command:**"))
            display(Markdown(f"`{export_command}`"))
            ! $export_command
            print(f"✅ INT4 {model_id.value} model converted and can be found in {int4_model_dir}")


.. parsed-literal::

    ✅ INT4 llama-3.1-8b-instruct model already converted and can be found in llama-3.1-8b-instruct/INT8_compressed_weights


Let’s compare model size for different compression types

.. code:: ipython3

    fp16_weights = fp16_model_dir / "openvino_model.bin"
    int8_weights = int8_model_dir / "openvino_model.bin"
    int4_weights = int4_model_dir / "openvino_model.bin"

    if fp16_weights.exists():
        print(f"Size of FP16 model is {fp16_weights.stat().st_size / 1024 / 1024:.2f} MB")
    for precision, compressed_weights in zip([8, 4], [int8_weights, int4_weights]):
        if compressed_weights.exists():
            print(f"Size of model with INT{precision} compressed weights is {compressed_weights.stat().st_size / 1024 / 1024:.2f} MB")
        if compressed_weights.exists() and fp16_weights.exists():
            print(f"Compression rate for INT{precision} model: {fp16_weights.stat().st_size / compressed_weights.stat().st_size:.3f}")


.. parsed-literal::

    Size of model with INT4 compressed weights is 4435.75 MB


Select device for inference and model variant
'''''''''''''''''''''''''''''''''''''''''''''



   **Note**: There may be no speedup for INT4/INT8 compressed models on
   dGPU.

.. code:: ipython3

    import openvino as ov

    core = ov.Core()

    support_devices = core.available_devices
    if "NPU" in support_devices:
        support_devices.remove("NPU")

    device = widgets.Dropdown(
        options=support_devices + ["AUTO"],
        value="CPU",
        description="Device:",
        disabled=False,
    )

    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'GPU.0', 'GPU.1', 'AUTO'), value='CPU')



The cell below demonstrates how to instantiate model based on selected
variant of model weights and inference device

.. code:: ipython3

    available_models = []
    if int4_model_dir.exists():
        available_models.append("INT4")
    if int8_model_dir.exists():
        available_models.append("INT8")
    if fp16_model_dir.exists():
        available_models.append("FP16")

    model_to_run = widgets.Dropdown(
        options=available_models,
        value=available_models[0],
        description="Model to run:",
        disabled=False,
    )

    model_to_run




.. parsed-literal::

    Dropdown(description='Model to run:', options=('INT4',), value='INT4')



Instantiate pipeline with OpenVINO Generate API
-----------------------------------------------



`OpenVINO Generate
API <https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md>`__
can be used to create pipelines to run an inference with OpenVINO
Runtime.

Firstly we need to create pipeline with ``LLMPipeline``. ``LLMPipeline``
is the main object used for decoding. You can construct it straight away
from the folder with the converted model. It will automatically load the
``main model``, ``tokenizer``, ``detokenizer`` and default
``generation configuration``. We will provide directory with model and
device for ``LLMPipeline``. After that we will configure parameters for
decoding. We can get default config with ``get_generation_config()``,
setup parameters and apply the updated version with
``set_generation_config(config)`` or put config directly to
``generate()``. It’s also possible to specify the needed options just as
inputs in the ``generate()`` method, as shown below. Then we just run
``generate`` method and get the output in text format. We do not need to
encode input prompt according to model expected template or write
post-processing code for logits decoder, it will be done easily with
LLMPipeline.

.. code:: ipython3

    from transformers import AutoTokenizer
    from openvino_tokenizers import convert_tokenizer
    from openvino_genai import LLMPipeline

    if model_to_run.value == "INT4":
        model_dir = int4_model_dir
    elif model_to_run.value == "INT8":
        model_dir = int8_model_dir
    else:
        model_dir = fp16_model_dir
    print(f"Loading model from {model_dir}\n")

    # optionally convert tokenizer if used cached model without it
    if not (model_dir / "openvino_tokenizer.xml").exists() or not (model_dir / "openvino_detokenizer.xml").exists():
        hf_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        ov_tokenizer, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
        ov.save_model(ov_tokenizer, model_dir / "openvino_tokenizer.xml")
        ov.save_model(ov_tokenizer, model_dir / "openvino_detokenizer.xml")


    pipe = LLMPipeline(str(model_dir), device.value)
    print(pipe.generate("The Sun is yellow bacause", temperature=1.2, top_k=4, do_sample=True, max_new_tokens=10))


.. parsed-literal::

    Loading model from llama-3.1-8b-instruct/INT4_compressed_weights

     it is a giant ball of hot glowing gases.


Run Chatbot
-----------



Now, when model created, we can setup Chatbot interface using
`Gradio <https://www.gradio.app/>`__. The diagram below illustrates how
the chatbot pipeline works

.. figure:: https://user-images.githubusercontent.com/29454499/255523209-d9336491-c7ba-4dc1-98f0-07f23743ce89.png
   :alt: generation pipeline

   generation pipeline

As can be seen, the pipeline very similar to instruction-following with
only changes that previous conversation history additionally passed as
input with next user question for getting wider input context. On the
first iteration, it is provided instructions joined to conversation
history (if exists) converted to token ids using a tokenizer, then
prepared input provided to the model. The model generates probabilities
for all tokens in logits format. The way the next token will be selected
over predicted probabilities is driven by the selected decoding
methodology. You can find more information about the most popular
decoding methods in this
`blog <https://huggingface.co/blog/how-to-generate>`__. The result
generation updates conversation history for next conversation step. It
makes stronger connection of next question with previously provided and
allows user to make clarifications regarding previously provided
answers. `More about that, please, see
here. <https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html>`__

To make experience easier, we will use `OpenVINO Generate
API <https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README>`__.
Firstly we will create pipeline with ``LLMPipeline``. ``LLMPipeline`` is
the main object used for decoding. You can construct it straight away
from the folder with the converted model. It will automatically load the
main model, tokenizer, detokenizer and default generation configuration.
After that we will configure parameters for decoding. We can get default
config with ``get_generation_config()``, setup parameters and apply the
updated version with ``set_generation_config(config)`` or put config
directly to ``generate()``. It’s also possible to specify the needed
options just as inputs in the ``generate()`` method, as shown below.
Then we just run ``generate`` method and get the output in text format.
We do not need to encode input prompt according to model expected
template or write post-processing code for logits decoder, it will be
done easily with ``LLMPipeline``.

| There are several parameters that can control text generation quality:
  \* ``Temperature`` is a parameter used to control the level of
  creativity in AI-generated text. By adjusting the ``temperature``, you
  can influence the AI model’s probability distribution, making the text
  more focused or diverse.
| Consider the following example: The AI model has to complete the
  sentence “The cat is \____.” with the following token probabilities:

::

   playing: 0.5
   sleeping: 0.25
   eating: 0.15
   driving: 0.05
   flying: 0.05

   - **Low temperature** (e.g., 0.2): The AI model becomes more focused and deterministic, choosing tokens with the highest probability, such as "playing."
   - **Medium temperature** (e.g., 1.0): The AI model maintains a balance between creativity and focus, selecting tokens based on their probabilities without significant bias, such as "playing," "sleeping," or "eating."
   - **High temperature** (e.g., 2.0): The AI model becomes more adventurous, increasing the chances of selecting less likely tokens, such as "driving" and "flying."

-  ``Top-p``, also known as nucleus sampling, is a parameter used to
   control the range of tokens considered by the AI model based on their
   cumulative probability. By adjusting the ``top-p`` value, you can
   influence the AI model’s token selection, making it more focused or
   diverse. Using the same example with the cat, consider the following
   top_p settings:

   -  **Low top_p** (e.g., 0.5): The AI model considers only tokens with
      the highest cumulative probability, such as “playing.”
   -  **Medium top_p** (e.g., 0.8): The AI model considers tokens with a
      higher cumulative probability, such as “playing,” “sleeping,” and
      “eating.”
   -  **High top_p** (e.g., 1.0): The AI model considers all tokens,
      including those with lower probabilities, such as “driving” and
      “flying.”

-  ``Top-k`` is an another popular sampling strategy. In comparison with
   Top-P, which chooses from the smallest possible set of words whose
   cumulative probability exceeds the probability P, in Top-K sampling K
   most likely next words are filtered and the probability mass is
   redistributed among only those K next words. In our example with cat,
   if k=3, then only “playing”, “sleeping” and “eating” will be taken
   into account as possible next word.
-  ``Repetition Penalty`` This parameter can help penalize tokens based
   on how frequently they occur in the text, including the input prompt.
   A token that has already appeared five times is penalized more
   heavily than a token that has appeared only one time. A value of 1
   means that there is no penalty and values larger than 1 discourage
   repeated
   tokens.https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html

Prepare text streamer to get results runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Load the ``detokenizer``, use it to convert token_id to string output
format. We will collect print-ready text in a queue and give the text
when it is needed. It will help estimate performance.

.. code:: ipython3

    import re
    import numpy as np
    from queue import Queue
    from openvino_genai import StreamerBase

    core = ov.Core()

    detokinizer_path = Path(model_dir, "openvino_detokenizer.xml")


    class TextStreamerIterator(StreamerBase):
        def __init__(self, tokenizer):
            super().__init__()
            self.tokenizer = tokenizer
            self.compiled_detokenizer = core.compile_model(detokinizer_path.as_posix())
            self.text_queue = Queue()
            self.stop_signal = None

        def __iter__(self):
            return self

        def __next__(self):
            value = self.text_queue.get()
            if value == self.stop_signal:
                raise StopIteration()
            else:
                return value

        def put(self, token_id):
            openvino_output = self.compiled_detokenizer(np.array([[token_id]], dtype=int))
            text = str(openvino_output["string_output"][0])
            # remove labels/special symbols
            text = re.sub("<.*>", "", text)
            self.text_queue.put(text)

        def end(self):
            self.text_queue.put(self.stop_signal)

Setup of the chatbot life process function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



``bot`` function is the entry point for starting chat. We setup config
here, collect history to string and put it to ``generate()`` method.
After that it’s generate new chatbot message and we add it to history.

.. code:: ipython3

    from uuid import uuid4
    from threading import Event, Thread

    pipe = LLMPipeline(str(model_dir), device.value)

    max_new_tokens = 256

    start_message = model_configuration["start_message"]
    history_template = model_configuration.get("history_template")
    current_message_template = model_configuration.get("current_message_template")


    def get_uuid():
        """
        universal unique identifier for thread
        """
        return str(uuid4())


    def convert_history_to_input(history):
        """
        function for conversion history stored as list pairs of user and assistant messages to tokens according to model expected conversation template
        Params:
          history: dialogue history
        Returns:
          history in token format
        """
        new_prompt = f"{start_message}"
        if history_template is None:
            for user_msg, model_msg in history:
                new_prompt += user_msg + "\n" + model_msg + "\n"
            return new_prompt
        else:
            new_prompt = "".join(["".join([history_template.format(num=round, user=item[0], assistant=item[1])]) for round, item in enumerate(history[:-1])])
            new_prompt += "".join(
                [
                    "".join(
                        [
                            current_message_template.format(
                                num=len(history) + 1,
                                user=history[-1][0],
                                assistant=history[-1][1],
                            )
                        ]
                    )
                ]
            )

        return new_prompt


    def default_partial_text_processor(partial_text: str, new_text: str):
        """
        helper for updating partially generated answer, used by default

        Params:
          partial_text: text buffer for storing previosly generated text
          new_text: text update for the current step
        Returns:
          updated text string

        """
        partial_text += new_text
        return partial_text


    text_processor = model_configuration.get("partial_text_processor", default_partial_text_processor)


    def bot(message, history, temperature, top_p, top_k, repetition_penalty):
        """
        callback function for running chatbot on submit button click

        Params:
          message: new message from user
          history: conversation history
          temperature:  parameter for control the level of creativity in AI-generated text.
                        By adjusting the `temperature`, you can influence the AI model's probability distribution, making the text more focused or diverse.
          top_p: parameter for control the range of tokens considered by the AI model based on their cumulative probability.
          top_k: parameter for control the range of tokens considered by the AI model based on their cumulative probability, selecting number of tokens with highest probability.
          repetition_penalty: parameter for penalizing tokens based on how frequently they occur in the text.
          active_chat: chat state, if true then chat is running, if false then we should start it here.
        Returns:
          message: reset message and make it ""
          history: updated history with message and answer from chatbot
          active_chat: if we are here, the chat is running or will be started, so return True
        """
        streamer = TextStreamerIterator(pipe.get_tokenizer())

        config = pipe.get_generation_config()
        config.temperature = temperature
        config.top_p = top_p
        config.top_k = top_k
        config.do_sample = temperature > 0.0
        config.max_new_tokens = max_new_tokens
        config.repetition_penalty = repetition_penalty

        # history = [['message', 'chatbot answer'], ...]
        history.append([message, ""])
        new_prompt = convert_history_to_input(history)

        stream_complete = Event()

        def generate_and_signal_complete():
            """
            genration function for single thread
            """
            global start_time
            pipe.generate(new_prompt, config, streamer)
            stream_complete.set()

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        partial_text = ""
        for new_text in streamer:
            partial_text = text_processor(partial_text, new_text)
            history[-1][1] = partial_text
            yield "", history, streamer


    def stop_chat(streamer):
        if streamer is not None:
            streamer.end()
        return None


    def stop_chat_and_clear_history(streamer):
        if streamer is not None:
            streamer.end()
        return None, None

.. code:: ipython3

    import gradio as gr

    chinese_examples = [
        ["你好!"],
        ["你是谁?"],
        ["请介绍一下上海"],
        ["请介绍一下英特尔公司"],
        ["晚上睡不着怎么办？"],
        ["给我讲一个年轻人奋斗创业最终取得成功的故事。"],
        ["给这个故事起一个标题。"],
    ]

    english_examples = [
        ["Hello there! How are you doing?"],
        ["What is OpenVINO?"],
        ["Who are you?"],
        ["Can you explain to me briefly what is Python programming language?"],
        ["Explain the plot of Cinderella in a sentence."],
        ["What are some common mistakes to avoid when writing code?"],
        ["Write a 100-word blog post on “Benefits of Artificial Intelligence and OpenVINO“"],
    ]

    japanese_examples = [
        ["こんにちは！調子はどうですか?"],
        ["OpenVINOとは何ですか?"],
        ["あなたは誰ですか?"],
        ["Pythonプログラミング言語とは何か簡単に説明してもらえますか?"],
        ["シンデレラのあらすじを一文で説明してください。"],
        ["コードを書くときに避けるべきよくある間違いは何ですか?"],
        ["人工知能と「OpenVINOの利点」について100語程度のブログ記事を書いてください。"],
    ]

    examples = chinese_examples if (model_language.value == "Chinese") else japanese_examples if (model_language.value == "Japanese") else english_examples


    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        streamer = gr.State(None)
        conversation_id = gr.State(get_uuid)
        gr.Markdown(f"""<h1><center>OpenVINO {model_id.value} Chatbot</center></h1>""")
        chatbot = gr.Chatbot(height=500)
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(
                    label="Chat Message Box",
                    placeholder="Chat Message Box",
                    show_label=False,
                    container=False,
                )
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit")
                    stop = gr.Button("Stop")
                    clear = gr.Button("Clear")
        with gr.Row():
            with gr.Accordion("Advanced Options:", open=False):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            temperature = gr.Slider(
                                label="Temperature",
                                value=0.1,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                interactive=True,
                                info="Higher values produce more diverse outputs",
                            )
                    with gr.Column():
                        with gr.Row():
                            top_p = gr.Slider(
                                label="Top-p (nucleus sampling)",
                                value=1.0,
                                minimum=0.0,
                                maximum=1,
                                step=0.01,
                                interactive=True,
                                info=(
                                    "Sample from the smallest possible set of tokens whose cumulative probability "
                                    "exceeds top_p. Set to 1 to disable and sample from all tokens."
                                ),
                            )
                    with gr.Column():
                        with gr.Row():
                            top_k = gr.Slider(
                                label="Top-k",
                                value=50,
                                minimum=0.0,
                                maximum=200,
                                step=1,
                                interactive=True,
                                info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                            )
                    with gr.Column():
                        with gr.Row():
                            repetition_penalty = gr.Slider(
                                label="Repetition Penalty",
                                value=1.1,
                                minimum=1.0,
                                maximum=2.0,
                                step=0.1,
                                interactive=True,
                                info="Penalize repetition — 1.0 to disable.",
                            )
        gr.Examples(examples, inputs=msg, label="Click on any example and press the 'Submit' button")

        submit_event = msg.submit(
            fn=bot,
            inputs=[msg, chatbot, temperature, top_p, top_k, repetition_penalty],
            outputs=[msg, chatbot, streamer],
            queue=True,
        )
        submit_click_event = submit.click(
            fn=bot,
            inputs=[msg, chatbot, temperature, top_p, top_k, repetition_penalty],
            outputs=[msg, chatbot, streamer],
            queue=True,
        )
        stop.click(fn=stop_chat, inputs=streamer, outputs=[streamer], queue=False)
        clear.click(fn=stop_chat_and_clear_history, inputs=streamer, outputs=[chatbot, streamer], queue=False)

    # if you are launching remotely, specify server_name and server_port
    #  demo.launch(server_name='your server name', server_port='server port in int')
    # if you have any issue to launch on your platform, you can pass share=True to launch method:
    # demo.launch(share=True)
    # it creates a publicly shareable link for the interface. Read more in the docs: https://gradio.app/docs/
    try:
        demo.launch()
    except Exception:
        demo.launch(share=True)


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7862

    To create a public link, set `share=True` in `launch()`.








.. code:: ipython3

    # please uncomment and run this cell for stopping gradio interface
    # demo.close()

Next Step
~~~~~~~~~



Besides chatbot, we can use LangChain to augmenting LLM knowledge with
additional data, which allow you to build AI applications that can
reason about private data or data introduced after a model’s cutoff
date. You can find this solution in `Retrieval-augmented generation
(RAG) example <llm-rag-langchain-with-output.html>`__.
