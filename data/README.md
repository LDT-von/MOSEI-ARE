# Data location and handling

Contest attachments remain outside this repository:

`../E题数据/E题数据/附件1-数据集原始多模态样本/`

`../E题数据/E题数据/附件2-数据集特征文件/`

`../E题数据/E题数据/附件3-模态缺失特征样本/`

`../E题数据/E题数据/附件4-可解释专项视频样本与特征文件/`

Keep the supplied directory structure and sample labels intact. If attachments move, record the new location in a local ignored config and regenerate the sample inventory and source hashes. Do not commit raw videos, full preprocessed pickle files or cached model weights.

The first-question `input_manifest.json` lists sample IDs, relative media paths and checksums. Rebuild it with `problem1/inventory.py` after changing the data location.

Keep the selected aligned or unaligned Attachment 2 feature release consistent across a problem's training, validation and corresponding special-test data. Record the selected version for each experiment.
