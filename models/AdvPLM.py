# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel
import os

def MixoutWrapper(module: nn.Module, p: float = 0.5):
    """
    Implementation of Mixout (https://arxiv.org/abs/1909.11299).
    Use with:
    >>> mixout_model = model.apply(MixoutWrapper).
    """
    # duplicate all the parameters, making copies of them and freezing them
    module._names = []
    module._params_orig = dict()
    _params_learned = nn.ParameterDict()
    for n, q in list(module.named_parameters(recurse=False)):
        c = q.clone().detach()
        c.requires_grad = False
        module._params_orig[n] = c
        _params_learned[n] = q
        module._names.append(n)
        delattr(module, n)
        setattr(module, n, c)
    if module._names:
        module._params_learned = _params_learned

    def mixout(module, n):
        if module.training:
            o = module._params_orig[n]
            mask = (torch.rand_like(o) < p).type_as(o)
            # update 2020-02-
            return (
                mask * module._params_orig[n]
                + (1 - mask) * module._params_learned[n]
                - p * module._params_orig[n]
            ) / (1 - p)
        else:
            return module._params_learned[n]

    def hook(module, input):
        for n in module._names:
            v = mixout(module, n)
            setattr(module, n, v)

    module.register_forward_pre_hook(hook)
    return module

class BertModel_modified(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if input_ids is not None:
            embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
            )
            return embedding_output

        elif inputs_embeds is not None:
            embedding_output = inputs_embeds
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

            assert(not return_dict)
            return (sequence_output, pooled_output) + encoder_outputs[1:]

class RobertaModel_modified(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if input_ids is not None:
            embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
            )
            return embedding_output

        elif inputs_embeds is not None:
            embedding_output = inputs_embeds
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

            assert(not return_dict)
            return (sequence_output, pooled_output) + encoder_outputs[1:]


class AdvPLM(nn.Module): 
    def __init__(self, opt):
        super(AdvPLM, self).__init__()
        self.opt=opt
        
        if opt.plm_type=="bert":
            self.plm = BertModel_modified.from_pretrained('bert-base-uncased')
            self.plm_teacher = BertModel_modified.from_pretrained('bert-base-uncased')

        elif opt.plm_type=="roberta":
            self.plm = RobertaModel_modified.from_pretrained('roberta-base')
            self.plm_teacher = RobertaModel_modified.from_pretrained('roberta-base')

        if opt.mixout_p>0:
            self.plm = MixoutWrapper(self.plm, opt.mixout_p)

        self.plm = torch.nn.DataParallel(self.plm)
        self.plm_teacher = torch.nn.DataParallel(self.plm_teacher)

        if opt.freeze_plm:
            for name, param in self.plm.named_parameters():
                param.requires_grad=False
        else:
            for name, param in self.plm.named_parameters():
                param.requires_grad=True

        if opt.freeze_plm_teacher:
            for name, param in self.plm_teacher.named_parameters():
                param.requires_grad=False
        else:
            for name, param in self.plm_teacher.named_parameters():
                param.requires_grad=True

        if self.opt.infonce_sim_metric == "normal":

            self.info_nonlinear = nn.Sequential(nn.Linear(768,512),nn.ReLU(),nn.Linear(512,256))
            self.info_nonlinear_teacher = nn.Sequential(nn.Linear(768,512),nn.ReLU(),nn.Linear(512,256))

            self.info_nonlinear_giveny_list = nn.ModuleList([])
            self.info_nonlinear_giveny_teacher_list = nn.ModuleList([])

            for y in range(self.opt.label_size):
                self.info_nonlinear_giveny_list.append(nn.Sequential(nn.Linear(768,512),nn.ReLU(),nn.Linear(512,256)))
                self.info_nonlinear_giveny_teacher_list.append(nn.Sequential(nn.Linear(768,512),nn.ReLU(),nn.Linear(512,256)))

            self.infonce_temperature = opt.infonce_temperature
        else:
            raise NotImplementedError

        self.cls_to_logit = nn.Sequential(nn.Linear(768,256), nn.ReLU(), nn.Linear(256,opt.label_size))

    def embd_to_logit(self, embd, attention_mask):

        _, pooled = self.plm(inputs_embeds=embd, attention_mask=attention_mask) 
        logits = self.cls_to_logit(pooled)

        return logits

    def embd_to_infonce(self, embd, embd_teacher, attention_mask, sim_metric, y):

        bs = (embd.shape)[0]
        _, pooled = self.plm(inputs_embeds=embd, attention_mask=attention_mask) 
        
        with torch.no_grad():
            _, pooled_teacher = self.plm_teacher(inputs_embeds=embd_teacher, attention_mask=attention_mask) 

        loss = 0
        
        if sim_metric=='normal':

            feature = pooled
            feature_teacher = pooled_teacher

            if y is None:
                feature = self.info_nonlinear(feature)
                feature_teacher = self.info_nonlinear_teacher(feature_teacher)
            else:
                feature = self.info_nonlinear_giveny_list[y](feature)
                feature_teacher = self.info_nonlinear_giveny_teacher_list[y](feature_teacher)

            feature = torch.nn.functional.normalize(feature)
            feature_teacher = torch.nn.functional.normalize(feature_teacher)

            bs = len(feature)
            f_a_b = torch.mm(feature, torch.transpose(feature_teacher, 0, 1) )#bs*768 * 768*bs = bs * bs

            f_a_b = f_a_b/self.infonce_temperature

            lsoftmax_0 = nn.LogSoftmax(0)
            loss = - torch.sum(torch.diag(lsoftmax_0(f_a_b)))/bs

        else:
            raise(NotImplementedError)

        return loss

    def text_to_embd(self, input_ids):
        embedding_output = self.plm(input_ids=input_ids)
        return embedding_output

    def text_to_embd_teacher(self, input_ids):
        embedding_output = self.plm_teacher(input_ids=input_ids)
        return embedding_output
    
    def ascc_attack(self, text_x, attention_mask, y, text_subs, text_subs_mask, ascc_attack_info):
        
        # record context
        plm_training_context = self.plm.training
        # set context
        self.plm.eval()
        self.cls_to_logit.eval()

        # text to embeddings
        with torch.no_grad():
            embd_x = self.text_to_embd(text_x) #input shape: bs, sent len, vocab size

        # substitution words to embeddings
        with torch.no_grad():
            n,l,s = text_subs.shape
            embd_subs = self.text_to_embd(text_subs.permute(0, 2, 1).reshape(n*s,l)).reshape(n,s,l,-1).permute(0,2,1,3)

        num_steps=ascc_attack_info['num_steps']
        loss_func=ascc_attack_info['loss_func']
        ascc_w_optm_lr=ascc_attack_info['ascc_w_optm_lr']
        sparse_weight = ascc_attack_info['sparse_weight']
        out_type = ascc_attack_info['out_type']

        device = embd_x.device
        batch_size, text_len, embd_dim = embd_x.shape
        batch_size, text_len, syn_num, embd_dim = embd_subs.shape

        hat_w = torch.empty(batch_size, text_len, syn_num, 1).to(device).to(embd_x.dtype)
        nn.init.kaiming_normal_(hat_w)
        hat_w.requires_grad_()
        params = [hat_w] 
        optimizer = torch.optim.Adam(params, lr=ascc_w_optm_lr, weight_decay=2e-5)

        def get_comb_w_logits(hat_w, text_subs_mask):
            w_logits=hat_w*text_subs_mask.reshape(batch_size, text_len, syn_num, 1) + 500*(text_subs_mask.reshape(batch_size, text_len, syn_num, 1)-1)
            return w_logits

        def get_comb(w, embd_subs):
            return (w* embd_subs.detach()).sum(-2)

        with torch.no_grad():
            logit_ori = self.embd_to_logit(embd_x.detach(), attention_mask)

        for _ in range(num_steps):
            optimizer.zero_grad()
            with torch.enable_grad():
                w_logits = get_comb_w_logits(hat_w, text_subs_mask)
                comb_w = F.softmax(w_logits, -2)
                embd_adv = get_comb(comb_w, embd_subs)

                if loss_func=='ce':
                    logit_adv = self.embd_to_logit(embd_adv, attention_mask)
                    loss = -F.cross_entropy(logit_adv, y, reduction='sum')
                elif loss_func=='kl':
                    logit_adv = self.embd_to_logit(embd_adv, attention_mask)
                    criterion_kl = nn.KLDivLoss(reduction="sum")
                    loss = -criterion_kl(F.log_softmax(logit_adv, dim=1),
                                        F.softmax(logit_ori.detach(), dim=1))
                                    
                if sparse_weight !=0:
                    loss_sparse = (-F.softmax(w_logits, -2)*F.log_softmax(w_logits, -2)).sum(-2).sum() / (300*128)
                    loss = loss + sparse_weight*loss_sparse

            loss.backward()
            optimizer.step()

        comb_w = F.softmax(get_comb_w_logits(hat_w, text_subs_mask), -2)

        if out_type == "text":
            comb_w = comb_w.reshape(batch_size* text_len, syn_num)
            ind = comb_w.max(-1)[1] # shape batch_size* text_len
            out = (text_subs.reshape(batch_size* text_len, syn_num)[np.arange(batch_size*text_len), ind]).reshape(batch_size, text_len)
        elif out_type == "adv_comb_w":
            out = comb_w

        # resume context
        if plm_training_context:
            self.plm.train()
            self.cls_to_logit.train()
        else:
            self.plm.eval()
            self.cls_to_logit.eval()

        return out.detach()

    def comb_w_to_logit(self, comb_w, text_subs, attention_mask):

        n, l, s = text_subs.shape
        embd_subs = self.text_to_embd(text_subs.permute(0, 2, 1).reshape(n*s,l)).reshape(n,s,l,-1).permute(0,2,1,3)
        embd = (comb_w*embd_subs).sum(-2)
        out = self.embd_to_logit(embd, attention_mask)

        return out

    def comb_w_to_mutual_info(self, text_x, comb_w, text_subs, attention_mask, sim_metric, y):

        n, l, s = text_subs.shape
        embd_subs = self.text_to_embd(text_subs.permute(0, 2, 1).reshape(n*s,l)).reshape(n,s,l,-1).permute(0,2,1,3)
        embd = (comb_w*embd_subs).sum(-2)
        embd_teacher = self.text_to_embd_teacher(text_x) 
        out = self.embd_to_infonce(embd, embd_teacher, attention_mask, sim_metric, y)

        return out

    def text_to_mutual_info(self, text_x, attention_mask, sim_metric, y):

        embd = self.text_to_embd(text_x)
        embd_teacher = self.text_to_embd_teacher(text_x)
        out = self.embd_to_infonce(embd, embd_teacher, attention_mask, sim_metric, y)

        return out

    def params_l2(self):
        loss = 1e-10
        for (param_name, param), (param_teacher_name, param_teacher) in zip(self.plm.named_parameters(),self.plm_teacher.named_parameters()):
            if param_name==param_teacher_name:
                loss += ((param-param_teacher)**2).sum()
        out = loss.sqrt()

        return out

    def forward(self, text_x, attention_mask):
       
        embd = self.text_to_embd(text_x)
        out = self.embd_to_logit(embd, attention_mask)

        return out