import glob
import os.path as osp

from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase, Datum_sf


@DATASET_REGISTRY.register()
class VLCS_SF(DatasetBase):
    """VLCS.

    Statistics:
        - 4 domains: CALTECH, LABELME, PASCAL, SUN
        - 5 categories: bird, car, chair, dog, and person.

    Reference:
        - Torralba and Efros. Unbiased look at dataset bias. CVPR 2011.
    """

    dataset_dir = "VLCS"
    domains = ["none", "caltech", "labelme", "pascal", "sun"]
    data_url = "https://drive.google.com/uc?id=1r0WL5DDqKfSPp9E3tRENwHaXNs1olLZd"

    def __init__(self, cfg, train_data):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "images")
        self.split_dir = osp.join(self.dataset_dir, "splits")

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "vlcs.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )
        test_datasets = []
        for domain in cfg.DATASET.TARGET_DOMAINS:
            test_datasets.append(self._read_data([domain], "test"))
        val_datasets = []
        for domain in cfg.DATASET.SOURCE_DOMAINS:
            val_datasets.append(self._read_data([domain], "crossval"))
        train = self._read_train_data(train_data, cfg.DATASET.SOURCE_DOMAINS, "train")

        super().__init__(train_x=train, val=val_datasets, test=test_datasets)

    def _read_train_data(self, train_data, input_domains, split):
        items = []
        classnames = train_data["classnames"]
        n_cls = train_data["n_cls"]
        n_style = train_data["n_style"]
        
        def _load_data_from_directory(directory):
            folders = listdir_nohidden(directory)
            folders.sort()
            items_ = []

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(directory, folder, "*.jpg"))

                for impath in impaths:
                    items_.append((impath, label))

            return items_
        
        for domain, dname in enumerate(input_domains):
            if split == "all":
                if dname.lower() == 'caltech':
                    actual_domain_folder = 'CALTECH'
                elif dname.lower() == 'pascal':
                    actual_domain_folder = 'PASCAL' 
                elif dname.lower() == 'labelme':
                    actual_domain_folder = 'LABELME'
                else:
                    actual_domain_folder = 'SUN'
                train_dir = osp.join(self.dataset_dir, actual_domain_folder, "train")
                impath_label_list = _load_data_from_directory(train_dir)
                val_dir = osp.join(self.dataset_dir, actual_domain_folder, "crossval")
                impath_label_list += _load_data_from_directory(val_dir)
            else:
                if dname.lower() == 'caltech':
                    actual_domain_folder = 'CALTECH'
                elif dname.lower() == 'pascal':
                    actual_domain_folder = 'PASCAL' 
                elif dname.lower() == 'labelme':
                    actual_domain_folder = 'LABELME'
                else:
                    actual_domain_folder = 'SUN'
                split_dir = osp.join(self.dataset_dir, actual_domain_folder, split)
                
                impath_label_list = _load_data_from_directory(split_dir)
        
            for impath, label in impath_label_list: # label=0
                classname = impath.split("/")[-2]
                for idx_style in range(n_style):
                    item = Datum_sf(
                        cls=label, # 0
                        style=idx_style,
                        label=label,
                        classname=classnames[label],
                        impath=impath,
                        domain=domain,
                    )
                    items.append(item)
                        
            '''
            for impath, label in impath_label_list: # label=0
                classname = impath.split("/")[-2]
                for idx_cls in range(n_cls):
                    for idx_style in range(n_style):
                        item = Datum_sf(
                            cls=idx_cls, # 0
                            style=idx_style,
                            label=idx_cls,
                            classname=classnames[idx_cls],
                            impath=impath,
                            domain=domain,
                        )
                        a = idx_cls
                        b = label
                        c = n_cls
                        d = classname
                        e = idx_style
                        f = n_style
                        g = impath
                        breakpoint()
                        items.append(item)
            '''
        
        return items
    
    def _read_data(self, input_domains, split):
        items = []
        ''' # code cũ không láy được CLASS label
        for domain, dname in enumerate(input_domains):
            dname = dname.upper()
            path = osp.join(self.dataset_dir, dname, split)
            folders = listdir_nohidden(path)
            folders.sort()

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(path, folder, "*.jpg"))

                for impath in impaths:
                    item = Datum(impath=impath, label=label, domain=domain)
                    items.append(item)
        '''
        def _load_data_from_directory(directory):
            folders = listdir_nohidden(directory)
            folders.sort()
            items_ = []

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(directory, folder, "*.jpg"))

                for impath in impaths:
                    items_.append((impath, label))

            return items_

        items = []
        for domain, dname in enumerate(input_domains):
            if split == "all":
                if dname.lower() == 'caltech':
                    actual_domain_folder = 'CALTECH'
                elif dname.lower() == 'pascal':
                    actual_domain_folder = 'PASCAL' 
                elif dname.lower() == 'labelme':
                    actual_domain_folder = 'LABELME'
                else:
                    actual_domain_folder = 'SUN'
                train_dir = osp.join(self.dataset_dir, actual_domain_folder, "train")
                impath_label_list = _load_data_from_directory(train_dir)
                val_dir = osp.join(self.dataset_dir, actual_domain_folder, "crossval")
                impath_label_list += _load_data_from_directory(val_dir)
            else:
                if dname.lower() == 'caltech':
                    actual_domain_folder = 'CALTECH'
                elif dname.lower() == 'pascal':
                    actual_domain_folder = 'PASCAL' 
                elif dname.lower() == 'labelme':
                    actual_domain_folder = 'LABELME'
                else:
                    actual_domain_folder = 'SUN'
                split_dir = osp.join(self.dataset_dir, actual_domain_folder, split)
                
                impath_label_list = _load_data_from_directory(split_dir)

            for impath, label in impath_label_list:
                class_name = impath.split("/")[-2].lower()
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=class_name
                )
                items.append(item)
        
        return items

    