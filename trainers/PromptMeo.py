import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
from torch.cuda.amp import GradScaler, autocast
from dassl.modeling import build_head, build_backbone
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DataManager, DataManager_sf
from dassl.evaluation import build_evaluator
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES # cần thay cái này thành các style template của mình
import os
from .style_generator import RandomStyleGenerator, BaseStyleGenerator, MixStyleGenerator, \
    RandomMixStyleGenerator, PromptStylerGenerator

# style_embedding # torch.Size([80, 1, 512])
# ta cần fixed_embeding # torch.Size([số_lớp, kích_thước_embedding]) = [n_cls, 512]
# task cần làm cần xem prompt_styler xử lí form: "a X style of cls" như nào 
# nếu theo đoán thì chắc họ style embed + cls_embed

exist = lambda target_path: os.path.exists(target_path)
_tokenizer = _Tokenizer()
'''

class clip_net(nn.Module):
    def __init__(self, cfg, model_cfg, device, **kwargs):
        super().__init__()

        self.device = device
        self.backbone = build_backbone( #vitb16clip lấy decorate từ file backbone clip
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            device=self.device,
            **kwargs
        )
        self.fdim = self.backbone._out_features
        self.head = None
        if model_cfg.HEAD.NAME == 'se_attn_sr':
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                num_channels=self.fdim,
                #reduction_ratio=model_cfg.HEAD.REDUCTION_RATIO,
                **kwargs,
            )
            self.fdim = self.head.out_features

    def forward_text(self, x, t_x):# forward_text(self, prompts, tokenized_prompts):
        t = self.backbone.forward_text(x, t_x)  # text embed without norm
        t = t / t.norm(dim=-1, keepdim=True)  # norm after embed
        if self.head is not None:
            t = self.head(t)  # text embed after head without norm
        return t

    def forward_img(self, x):  # for test
        t_img = self.backbone.forward_image(x)
        t_img = t_img / t_img.norm(dim=-1, keepdim=True)  # norm after embed
        if self.head is not None:
            t_img = self.head(t_img)  # img embed after head without norm
        return t_img

'''

def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    '''
    if (backbone_name=="vitb16_clip"):
        bb_name = 'ViT-B/16'
    elif (backbone_name=="resnet50_clip"):
        bb_name = 'RN50'
    elif (backbone_name=="vitl14_clip"):
        bb_name = 'ViT-L/14'
    '''    
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION,
                          "language_depth": cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT,
                          "vision_ctx": cfg.TRAINER.PROMPTSRC.N_CTX_VISION,
                          "language_ctx": cfg.TRAINER.PROMPTSRC.N_CTX_TEXT}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

#cần thêm hàm foward_text với load_weight model
class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg.TRAINER.PROMPTSRC.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PROMPTSRC.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PROMPTSRC.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # Also create frozen CLIP
        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg, True)
        #code thêm hàm tính fixed_embedding có tensor[n_cls, size_embed]
        
        #chắc phải chỉnh lại cái device 
        #self.embed_layer = clip_net(cfg, cfg.MODEL, cfg.device)        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            # Ta cần chỉnh all_teacher_features để tính theo style embed, chạy vòng for(style)(class)
            # Now pre-compute the frozen VL embeddings 
            all_teacher_features = []
            # Using multiple text templates to ensure textual diversity during training
            for single_template in IMAGENET_TEMPLATES: # tạo 1 file chứa các template cho phase 1 r thêm nó vào IMAGENET_TEMPLATES là xong
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x]) # torch.Size([7, 77])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda()) #(số_lớp, kích_thước_embedding) # torch.Size([7, 512])
                all_teacher_features.append(text_features.unsqueeze(1)) #(số_lớp, 1, kích_thước_embedding)

            
        # torch.Size([7, 512]) (n_cls, embed)
        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1) # (số_lớp, 1, kích_thước_embedding) --> (số_lớp, n_template, kích_thước_embedding) -> (số_lớp, kích_thước_embedding)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts
    
    def forward_text(self, abx ):
        return 
    
    def load_weight(self, abc):
        return
        


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts # torch.cat([clip.tokenize(p) for p in prompts]) shape=(n_cls, n_tkn) mã hóa toàn bộ prompt theo dạng prompt + class (với toàn bộ class) 
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts # torch.Size([7, 77]) #token hóa theo form "a photo of cls"
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner() # torch.Size([7, 77, 512]) (n_cls, n_tkn, dim_embed)
        # Compute the prompted image and text features
        text_features = self.text_encoder(prompts, tokenized_prompts) #torch.Size([7, 512]) (n_cls, dim_embed)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits
        logits = logit_scale * image_features @ text_features.t()
        if self.prompt_learner.training:
            # Now calculate the frozen pre-trained features
            fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                # Compute pre-trained frozen visual features
                zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()

            return F.cross_entropy(logits, #fixed_embeddings là fix-embedding text, cần chỉnh cách tính theo cách tính của mình
                                   label), text_features, fixed_embeddings, zero_shot_features, \
                   image_features, zero_shot_logits, logits
        else:
            return logits
        
# code thêm hàm tính fixed_embedding tại đây có các hàm load_weight, forward_text
@TRAINER_REGISTRY.register()
class PromptMeo(TrainerX):
    def __init__(self, cfg):
        #super().__init__()
        self.style_generator: BaseStyleGenerator = None
        self.num_classes = None
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        self.cfg = cfg
        
        #self.embed_layers = clip_net(cfg, cfg.MODEL, self.device)
        self.build_train_data()
        self.build_data_loader()
        self.build_model()
        self.clip_model = load_clip_to_cpu(self.cfg)
        self.clip_model_cuda = load_clip_to_cpu(self.cfg).cuda()
        
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = -np.inf
        
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTMEO.PREC in ["fp16", "fp32", "amp"]
    
    def load_weight(self, device="cpu"): # load lại các style embedding vào model
        assert exist(self.weight_save_path), "prompt style weight path not exist!"
        # load weight
        state_dict = torch.load(self.weight_save_path, map_location=device)
        self.style_embedding = state_dict["style_embedding"]
        self.style_embedding.requires_grad_(False)
        print(f"load weight from {self.weight_save_path}")
    
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        # chỉnh lại đường dẫn chuẩn
        self.weight_save_path = "/mnt/disk1/theanh28/DPStyler/PromptStyler/output/pacs/vitb16_clip/PS_re_train_style/seed1/checkpoint/model.pth"
        #os.path.join(cfg.TRAINER.PROMPTMEO.WEIGHT_DIR_PATH,
        #                                     cfg.TRAINER.PROMPTMEO.CHECK_POINT_NAME)
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTMEO.PREC == "fp32" or cfg.TRAINER.PROMPTMEO.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        #self.embed_layers = clip_net(cfg, cfg.MODEL, self.device)
        self.text_encoder = TextEncoder(clip_model=clip_model)
        self.load_weight()
        x = self.style_embedding.shape # torch.Size([80, 1, 512]) (n_style, 1, dim_embed)
        #breakpoint()

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        mean = cfg.TRAINER.PROMPTMEO.GPA_MEAN
        stdev = cfg.TRAINER.PROMPTMEO.GPA_STD
        gauss = self.get_gauss(mean, stdev)
        self.gauss = np.array([gauss(a) for a in range(1, N + 1)])
        self.gauss = self.gauss / sum(self.gauss)
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTMEO.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        # Keep model with GPA
        self.previous_model_gpa = None

    def forward_backward(self, batch):
        image, label, input_stylized_embedding, input_tokenized_base_text = self.parse_batch_train(batch)
        # input_stylized_embedding #torch.Size([4, 77, 512])
        # input_tokenized_base_text # torch.Size([4, 77])
        model = self.model
        optim = self.optim
        scaler = self.scaler
        # sự phụ thuộc vào dim0
        # normalized_text_features, zs_clip_text_embeddings: #torch.Size([7, 512]) (phụ thuộc vào số class)
        # input_stylized_embedding, input_tokenized_base_text, image_ft: #torch.Size([4, *, 512]) (phụ thuộc vào batchsize)
        prec = self.cfg.TRAINER.PROMPTMEO.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
            zero_shot_logits, logits = model(image, label)
            ### vòng for(cls) {for(n_style) sau đó lấy mean}
            style_position = [1 for _ in self.classnames]
            #clip_model = load_clip_to_cpu(self.cfg)
            classnames = self.dm.dataset.classnames
            base_prompt = ["a photo of a" + " " + name + "." for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in base_prompt])  # (n_cls, n_tkn)
            base_embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.clip_model.dtype) #torch.Size([7, 77, 512])
            #base_embedding = clip_model_temp.encode_text(tokenized_prompts.cuda()) 
            #x = base_embedding.shape
            #print(f"classnames: {self.classnames}") #classnames: ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
            #y = classnames
            #breakpoint()
            # cần token và embed base_prompt
            # self.style_embedding # torch.Size([80, 1, 512])
            all_feature = [] 
            for index, name in enumerate(self.classnames):
                x = [[] for _ in range(len(self.classnames))]
                for style_id in range(self.cfg.TRAINER.NUM_STYLES): # 80
                    '''
                    #new_style_embedding = base_embedding.clone()
                    new_style_embedding = base_embedding[index].clone()
                    # ghi đè style_embed vào vị trí style_position trong base_embedding
                    #new_style_embedding[0, style_position:style_position + 1, :] = self.style_embedding[style_id].clone() 
                    new_style_embedding[index, style_position:style_position + 1, :] = self.style_embedding[style_id].clone()  
                    '''
                    #
                    stylized_embedding = self.get_stylized_embedding( #torch.Size([1, 77, 512])
                        base_embedding[index:index + 1, ...],
                        style_position[index], style_id)
                    #
                    x[index].append(stylized_embedding)
                # Shape kết quả: [80, 1, 77, 512]
                stacked_tensor = torch.stack(x[index], dim=0)
                # Shape [80, 1, 77, 512] -> [1, 77, 512]
                avg_tensor = torch.mean(stacked_tensor, dim=0)
                all_feature.append(avg_tensor)
            hamo_text_feat = torch.cat(all_feature, dim=0) #torch.Size([7, 77, 512])
            #a = hamo_text_feat.shape
            #tokenized_prompts
            # Kiểm tra hamo_text_feat
            #print(f"hamo_text_feat is on: {hamo_text_feat.device}")

            # Kiểm tra tokenized_prompts
            #print(f"tokenized_prompts is on: {tokenized_prompts.device}")
            #breakpoint()
            #text_encoder_output = self.embed_layers.forward_text(hamo_text_feat.cuda(), tokenized_prompts.cuda())
            #clip_model1 = load_clip_to_cpu(self.cfg).cuda()
            text_encoder_output = self.clip_model_cuda.forward_text(hamo_text_feat.cuda(), tokenized_prompts.cuda())
            #b = text_encoder_output.shape
            #breakpoint() 
            #new_style_embedding = base_embedding.clone()
            #new_style_embedding[0, style_position:style_position + 1, :] = self.style_embedding[style_id].clone()
            
            '''
            def get_stylized_embedding(self, single_base_embedding, style_position, style_id):
                assert style_id < len(self.style_embedding), "Style id is outside the length of the style list!"
                new_style_embedding = single_base_embedding.clone()
                new_style_embedding[0, style_position:style_position + 1, :] = self.style_embedding[style_id].clone()
                return new_style_embedding
                
            stylized_embedding = self.style_generator.get_stylized_embedding(
                self.base_embedding[item.label:item.label + 1, ...],
                self.style_position[item.label],
                item.style)
                
            self.tokenized_base_text = torch.cat([clip.tokenize(p) for p in base_text_list]).to(self.device)
            # torch.size([7, 77, 512])
            self.base_embedding = clip_model.token_embedding(self.tokenized_base_text).to("cpu") # torch.size([7, 77, 512])
            
            self.style_position = [1 for _ in self.classnames]
            '''
            
            ###
            #cal fixed text embed
            # yêu cầu torch.size[]
            #a = input_stylized_embedding.shape
            #b = input_tokenized_base_text.shape
            #c = image_ft.shape
            #breakpoint()
            #text_encoder_output_1 = self.text_encoder.forward(input_stylized_embedding, input_tokenized_base_text)
            #text_encoder_output = self.embed_layers.forward_text(input_stylized_embedding, input_tokenized_base_text)
            text_encoder_output = text_encoder_output / text_encoder_output.norm(dim=-1, keepdim=True)
            # Calculate the L_SCL_text loss
            #loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(),
            #                         reduction='mean') * self.cfg.TRAINER.PROMPTMEO.TEXT_LOSS_WEIGHT
            #a = normalized_text_features.shape #torch.Size([7, 512]) (n_cls, dim_embed)
            #b = text_encoder_output.shape # torch.Size([4, 512])
            #c = text_encoder_output_1.shape
            #breakpoint() 
            loss_scl_text = F.l1_loss(normalized_text_features, text_encoder_output.cuda(),
                                      reduction='mean') * self.cfg.TRAINER.PROMPTMEO.TEXT_LOSS_WEIGHT
            # Calculate the L_SCL_image loss
            loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(),
                                       reduction='mean') * self.cfg.TRAINER.PROMPTMEO.IMAGE_LOSS_WEIGHT
            # Now calculate L_SCL_logits
            L_SCL_logits = F.kl_div(
                F.log_softmax(logits / 1, dim=1),
                F.log_softmax(zero_shot_logits / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / logits.numel()
            L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)
            loss = (loss_ce + L_SCL)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            # Means one epoch is completed, perform GPA
            self.step_counter = self.step_counter + 1
            current_epoch_weight = self.gauss[self.step_counter - 2]
            current_model_weights = copy.deepcopy(model.state_dict())
            weighted_state_dict = self.state_dict_weighting(current_model_weights, current_epoch_weight)
            if self.previous_model_gpa is None:
                self.previous_model_gpa = weighted_state_dict
            else:
                self.previous_model_gpa = self.state_dict_add(weighted_state_dict, self.previous_model_gpa)

        if self.step_counter == self.model.total_epochs + 1:
            print("Using GPA model for final inference...")
            model.load_state_dict(self.previous_model_gpa)
            self.model.load_state_dict(self.previous_model_gpa)
        return loss_summary
    
    def build_data_loader(self):
        train_data = self.style_generator.train_data()
        dm = DataManager_sf(self.cfg, train_data)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}
        self.dm = dm
        
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input_stylized_embedding = batch["stylized_embedding"]
        input_tokenized_base_text = batch["tokenized_base_text"]
        input = input.to(self.device)
        label = label.to(self.device)
        input_stylized_embedding = input_stylized_embedding.to(self.device)
        input_tokenized_base_text = input_tokenized_base_text.to(self.device)
        return input, label, input_stylized_embedding, input_tokenized_base_text


    def state_dict_weighting(self, main_dict, weightage, prompt_only=False):
        # Average all parameters
        updated_dict = copy.deepcopy(main_dict)
        if not prompt_only:
            for key in main_dict:
                updated_dict[key] = main_dict[key] * weightage
            return updated_dict
        else:
            return main_dict * weightage

    def state_dict_add(self, dict1, dict2, prompt_only=False):
        # Average all parameters
        if not prompt_only:
            modified_dict = dict2
            for key in dict1:
                modified_dict[key] = (modified_dict[key] + dict1[key])
            return modified_dict
        else:
            return dict1 + dict2

    def get_gauss(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return gauss

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            
    def build_train_data(self):

        txts_dir_path = self.cfg.TXTS_PATH
        txt_path = os.path.join(txts_dir_path, self.cfg.DATASET.NAME + '.txt')

        with open(txt_path, 'r') as f:
            lines = f.read().splitlines()
        class_dict = {index: value for index, value in enumerate(lines)}
        classnames = list(class_dict.values())
        self.classnames = classnames
        self.num_classes = len(classnames)
        assert self.cfg.STYLE_GENERATOR.NAME in globals()
        clip_model = load_clip_to_cpu(self.cfg).cuda()
        #self.style_generator = globals()[self.cfg.STYLE_GENERATOR.NAME](self.cfg, classnames, #self.cfg.STYLE_GENERATOR.NAME = PromptStylerGenerator
        #                                                                self.embed_layers.backbone, self.device)
        #print(f"hamo_text_feat is on: {self.embed_layers.backbone.device}")

            # Kiểm tra tokenized_prompts
        #print(f"tokenized_prompts is on: {self.device}")
        #breakpoint()
        self.style_generator = globals()[self.cfg.STYLE_GENERATOR.NAME](self.cfg, classnames, #self.cfg.STYLE_GENERATOR.NAME = PromptStylerGenerator
                                                                        clip_model, self.device)
        #self.style_generator = 
        
    def train(self):
        self.before_train()
        self.style_generator.reinit_style()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.run_epoch()
            if self.epoch % 2 == 0:
                self.after_epoch()
        self.after_train()
        
    def get_stylized_embedding(self, single_base_embedding, style_position, style_id):
        assert style_id < len(self.style_embedding), "Style id is outside the length of the style list!"
        new_style_embedding = single_base_embedding.clone()
        #new_style_embedding[:, style_position:style_position + 1, :] = self.style_embedding[style_id].clone()
        new_style_embedding[0, style_position:style_position + 1, :] = self.style_embedding[style_id].clone()
        return new_style_embedding