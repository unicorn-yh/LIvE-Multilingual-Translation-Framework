

<img src="figure/title.png" style="zoom:67%;" />





## Introduction1

The **Linguistic Intermediary via English (LIvE)** framework leverages English as a pivot language in a Multilingual Neural Machine Translation (MNMT) system to enhance translation between low-resource languages, particularly in scientific domains like Neurobiology. Built on Google’s **GEMMA 2-9B** model, the framework employs supervised fine-tuning with LoRA for efficiency, following a three-stage process to optimize translations and develop direct low-resource language pair models. The framework follows a three-stage process: translating from a source low-resource language to English, refining translations from English to a target low-resource language, and developing a direct MNMT model for source-to-target translation.  With a curated domain-specific dataset and synthetic data, LIvE achieves superior performance, outperforming state-of-the-art models with hundreds of billions of parameters, showcasing its potential to democratize access to scientific knowledge and adapt to diverse low-resource languages and domains.



## Case Study

<img src="figure/case1.png" style="zoom:67%;" />

The figure above ilustrates a comparative evaluation of translations produced by various models using English as a pivot language to translate from Malay to Chinese. Corect translations preserve the scientific terminology and context from the original sentence, such as "电化学梯度" (electrochemical gradient) and "钠离子的平衡电位" (equilibrium potential of sodium ions). Errors are marked in red, highlighting deviations like substituting "电化学梯度" with "电解质梯度"(electrolyte gradient), which misrepresents the scientific meaning in terms of Neurobiology. The highlighted translations from GPT-4 and GLM-4 demonstrate high fluency but occasionally lose precision, while Google Translate maintains accuracy but lacks natural readability. **LIvE (Ours)** achieves a balanced translation. ensuring both fluency and fidelity to the original scientific context.
