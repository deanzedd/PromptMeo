import torch
import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset
from torch import nn as nn
import clip


from dassl.utils import read_image

from .datasets import build_dataset, build_dataset_sf
from .samplers import build_sampler
from .transforms import INTERPOLATION_MODES, build_transform


def build_data_loader(
        cfg,
        sampler_type="SequentialSampler",
        data_source=None,
        batch_size=64,
        n_domain=0,
        n_ins=2,
        tfm=None,
        is_train=True,
        dataset_wrapper=None,
        train_data=None,
        num_workers=0
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train, traindata=train_data),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
    assert len(data_loader) > 0

    return data_loader


class DataManager_sf:

    def __init__(
            self,
            cfg,
            train_data,
            custom_tfm_train=None,
            custom_tfm_test=None,
            dataset_wrapper=None,
            test_num_worker=8
    ):
        # Load dataset
        dataset = build_dataset_sf(cfg, train_data)
        self.train_data = train_data
        
        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train
        
        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x

        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=DatasetWrapper_train_sf,
            train_data=self.train_data,
            num_workers=cfg.DATALOADER.NUM_WORKERS
        )

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=DatasetWrapper_train_sf,
                train_data=self.train_data
            )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=DatasetWrapper_train_sf,
                train_data=self.train_data
            )
        
        val_loader_list = []
        for dataset_domain in dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset_domain,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper,
                num_workers=test_num_worker
            )
            val_loader_list.append(val_loader)
        
        test_loader_list = []
        # Build test_loader
        for dataset_domain in dataset.test:
            test_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset_domain,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper,
                num_workers=test_num_worker
            )
            test_loader_list.append(test_loader)
        
        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader_list
        self.test_loader = test_loader_list

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME
        source_domains = cfg.DATASET.SOURCE_DOMAINS
        target_domains = cfg.DATASET.TARGET_DOMAINS

        table = []
        table.append(["Dataset", dataset_name])
        if source_domains:
            table.append(["Source", source_domains])
        if target_domains:
            table.append(["Target", target_domains])
        table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# train_x", f"{len(self.dataset.train_x):,}"])
        if self.dataset.train_u:
            table.append(["# train_u", f"{len(self.dataset.train_u):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        for i in range(len(self.dataset.test)):
            table.append(["# test " + str(i), f"{len(self.dataset.test[i]):,}"])

        print(tabulate(table))


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False, traindata=None):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img

class BaseStyleGenerator(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        # a {} style of a {cls}
        self.classnames = classnames
        self.device = device
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.n_style = cfg.TRAINER.NUM_STYLES
        self.clip_model = clip_model
        self.enable_diverse = cfg.STYLE_GENERATOR.ENABLE_DIVERSE
        # Xây dựng vectơ đặc trưng cơ bản
        # Xây dựng một vectơ phong cách cơ bản cho mỗi loại
        base_text_list = ["a random style of a " + s for s in classnames]

        # position_offset = [0 if len(j.split("_")) == 1 else 2 for j in self.classnames]
        self.style_position = [1 for _ in self.classnames]
        # 基础的风格向量
        # torch.size([7, 77])
        self.tokenized_base_text = torch.cat([clip.tokenize(p) for p in base_text_list]).to(self.device)
        # torch.size([7, 77, 512])
        self.base_embedding = clip_model.token_embedding(self.tokenized_base_text).to("cpu")  # 将基础风格的token转为embedding
        self.style_embedding = []  # 保存k个风格
        self.stylized_base_text_encoder_out = []  # 保存只包含k个风格的文本（没有类别）

    def style_generator(self, embedding_dim=512):
        raise NotImplementedError("You must implement this function!")

    def get_stylized_embedding(self, single_base_embedding, style_position, style_id):
        assert style_id < len(self.style_embedding), "Style id is outside the length of the style list!"
        new_style_embedding = single_base_embedding.clone()
        new_style_embedding[0, style_position:style_position + 1, :] = self.style_embedding[style_id].clone()
        return new_style_embedding

    def _init_stylized_text(self, base_text=None, style_position=0):
        if base_text is None:
            base_text = "X-like style"
            style_position = 1
        base_text_list = [base_text] * self.n_style
        tokenized_base_text = torch.cat([clip.tokenize(p) for p in base_text_list]).to(self.device)
        stylized_base_text_embedding = self.clip_model.token_embedding(tokenized_base_text)  # 将基础风格的token转为embedding
        stylized_base_text_embedding[:, style_position:style_position + 1, :] = self.style_embedding
        self.stylized_base_text_encoder_out = self.clip_model.forward_text(stylized_base_text_embedding,
                                                                           tokenized_base_text).to("cpu")

    def reinit_style(self, embedding_dim=512):
        self.generate_style_embedding(embedding_dim)
        self._init_stylized_text()

    @torch.no_grad()
    def generate_style_embedding(self, embedding_dim=512):
        self.style_embedding = torch.cat([self.style_generator(embedding_dim) for _ in range(self.n_style)]).unsqueeze(
            1).to("cpu")

    def train_data(self):
        '''
        text_int:x
        text_init_tokenized:t_x
        "style_prompt":n_style种随机风格
        '''
        train_data = {"classnames": self.classnames,
                      "base_embedding": self.base_embedding.to("cpu"),
                      "tokenized_base_text": self.tokenized_base_text.to("cpu"),
                      "style_generator": self,
                      "n_cls": len(self.classnames),
                      "n_style": self.cfg.TRAINER.NUM_STYLES,
                      "style_position": self.style_position,
                      }
        return train_data


class DatasetWrapper_train_sf(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False, traindata=None):
        #from ..trainers.style_generator import BaseStyleGenerator

        self.cfg = cfg
        self.data_source = data_source
        self.traindata = traindata
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        # init
        self.style_position = self.traindata["style_position"]
        self.base_embedding = self.traindata["base_embedding"]
        self.style_generator: BaseStyleGenerator = self.traindata["style_generator"]
        self.tokenized_base_text = self.traindata["tokenized_base_text"]
        
        
        # Build transform that doesn't apply any data augmentation
        # transform cho ảnh
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "style": item.style,
            "classname": item.classname,
            "index": idx
        }
        
        # augument ảnh
        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation
        
        
        # tính các embed của stylized
        stylized_embedding = self.style_generator.get_stylized_embedding(
            self.base_embedding[item.label:item.label + 1, ...],
            self.style_position[item.label], item.style)
        # class_init_embedding = self.text_init_embedding[item.label:item.label + 1, :, :].clone()  # 1*77*512
        # style_init_embedding = self.style_prompt_vectors[item.style:item.style + 1, :, :]
        # class_init_embedding[:, self.class_token_position[item.label]:self.class_token_position[item.label] + 1,
        # :] = style_init_embedding  # 替换风格

        output["stylized_embedding"] = stylized_embedding.squeeze(0).detach() #torch.Size([4, 77, 512])
        output["tokenized_base_text"] = self.tokenized_base_text[item.label:item.label + 1, :].clone().squeeze(0).detach() # torch.Size([4, 77])

        return output
    
    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img

