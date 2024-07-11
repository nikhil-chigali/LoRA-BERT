Retrieval-Augmented Mixture of LoRA experts for Uploadable Machine Learning
===
The authors propose a framework to compose various LoRA's from repositories like HuggingFace. Depending on the downstream task given, their framework implements `LoRARetriever` that retrieves suitable LoRA modules, which are composed together and the input is routed through these modules using a `RouterLoRA` module which uses cross-attention mechanism. Lastly, to train the `LoRARetriever` and the `RouterLoRA` module using batch inputs, they implement a batch inference masking mechanism to accomodate heterogenous requests.

Their framework excels at generalizing to unseen LoRA modules that are continuously being added to the repository. The modules - `LoRARetriever` and `RouterLoRA` are trained using *Instruction tuning* and have shown good zero-shot capabilities on the unseen LoRAs.

## 1. Input-aware LoRA Retrieval [LoRARetriever]
Inspired from RAG models that use similarity search to retrieve relevant context from the vector stores. Here, they create a task-specific LoRAs embedding space.
- Each LoRA module is embedded with `m` randomly selected domain-specific samples,
> $E(\phi) = \frac{1}{m}\sum_{i=1}^{m}E(I\bigoplus x_{i\phi})$ \
> where,\
> $\phi$ : LoRA being embedded \
> $I$ : Instruction (`Represent the sentence for similar task retrieval`) \
> $x_{i\phi}$ : Sample `i` from $LoRA_{\phi}$
- Input sentence `x` is also encoded in a similar manner,
> $E(\phi) = E(I\bigoplus x)$
- Cosine similarity is leveraged for measuring the similarity between the embeddings, 
> $s(x, \phi, I) = cos(E(I \bigoplus x), E(\phi))$

The LoRARetriever is then trained using contrastive loss.

## 2. On-the-fly Mixture of LoRA Experts\*
Builds and improves on the existing paper Mixture of LoRA Experts - https://arxiv.org/abs/2404.13628. They implement a cross-attention mechanism to assign attention weights to each LoRA from the previous step.\
Let, $A = \{A_1, A_2, \dots, A_n\}$, $B = \{B_1, B_2, \dots, B_n\}$ be the LoRA sets of top-n LoRAs from the `LoRARetriever`.
$A_r, B_r$ be the params of the `RouterLoRA`

- Cross-attention: Compares input $x$ with each $\phi_i$ to assign weights to each LoRA
> Value : $v_i = B_iA_ix$ \
> Query : $q = A_rx$ \
> Key : $k_i = B_r^Tv_i$ \
> $s_i = <q, k_i>/\sqrt{r}$ \
> Attention : $\alpha = Softmax(s_1, s_2, \dots, s_n)$ \
> $x\prime = \sum_{i=1}^{k}\alpha_iv_i$

- Random Dropout is used on LoRA modules for improving zero-shot generalization on OOD(out-of-distribution) scenarios.

## 3. Batch Inference of Multiple LoRAs
For batch of inputs $X\in \mathbb{R}^{b \times l \times d}$ where `b` is batch size, `l` is sequence length, and `d` is sample dimensionality, `b` sets of LoRAs are retrieved, $\Phi_B = \{\Phi_1, \Phi_2, \dots, \Phi_b\}$, where each set - $\Phi_i$ has top-k retrieved LoRAs for that input $x_i$
- Firstly, they create a collective set of all the LoRA sets by de-duplicating overlapping LoRAs resulting in a set $\Phi_B$ with $p \leq bk$ unique LoRAs
- For every sample $x_i$, a `p` dimension mapping vector $M_i$ is generated which specifies the indices of its corresponding LoRAs within $\Phi_B$
- The LoRA mapping vectors are combined into a matrix $M\in \mathbb{R}^{b\times p}$
- Using this mask, respective LoRA weights are gathered into a single matrix for efficient batch processing.


### My Thoughts:
- The authors introduce a framework that composes different LoRAs for multi-task inputs. This generalizes well for new unseen LoRAs using instruction-tuning, and regularization methods(like random lora dropout for IID examples).
- Step 2 (On-the-fly MoLE) is what interests me the most. They use Cross attention to control the routing to relevant LoRAs and decouple the irrelevant ones. How this is different from what we are trying to achieve is that they keep each individual LoRA separate and compose them using a separate `RouterLoRA` module. 
- I am interested to see if LoRAs form different tasks come together to form a unique subspace that generalizes well to all these tasks. However, this could be challenging as it's highly dependant on the nature of the task and its LoRA subspace. 
- I kind of understand how orthogonality between LoRAs can be a good lead as orthogonal low-rank adapters joined to form a new subspace and if this new subpsace would perform on these individual tasks.