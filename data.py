from torch.utils.data import Dataset

import DataUtil
import config
import config as cf
import utils

import os



import gc

class CodePtrDataset(Dataset):


    def __init__(self, mode):
        assert mode in ['train', 'valid', 'test']  

        sources, codes, sliced_flatten_ast, asts_vocabs, nls = [], [], [], [], []
        sliced_flatten_ast_datas = DataUtil.read_pickle_data(
            os.path.join(cf.ASTs_dir, "sliced_flatten_ast_listEdge.pkl"))
        useful_fids = DataUtil.read_pickle_data(os.path.join(cf.ASTs_dir, "sliced_flatten_ast_fid.pkl"))
        sliced_flatten_ast_datas = DataUtil.read_pickle_data(os.path.join(cf.ASTs_dir, "sliced_flatten_ast_listEdge.pkl"))
        asts_vocab_datas = DataUtil.read_pickle_data(os.path.join(cf.ASTs_dir, "ast_vocab_input.pkl"))

        soucrce_comments = DataUtil.read_pickle_data(os.path.join(cf.code_summary_dir, "source_comment1.pkl"))
        token_sources = DataUtil.read_pickle_data(os.path.join(cf.token_dir, "token_source.pkl"))
        token_codes = DataUtil.read_pickle_data(os.path.join(cf.token_dir, "token_code.pkl"))
        i = 0
        for fid in useful_fids[mode]:

            sliced_flatten_ast.append(sliced_flatten_ast_datas[fid])
            sources.append(token_sources[fid].lower().strip().split(' '))
            codes.append(token_codes[fid].lower().strip().split(' '))
            nls.append(soucrce_comments[fid].lower().strip().split(' '))
            asts_vocabs.append(asts_vocab_datas[fid])
            i = i+1

        del sliced_flatten_ast_datas
        del token_sources
        del token_codes
        del asts_vocab_datas
        del soucrce_comments
        gc.collect()


        assert len(sources) == len(codes) == len(nls) == len(asts_vocabs) == len(sliced_flatten_ast) 


        sources, codes, asts_vocabs, nls, sliced_flatten_ast = utils.filter_data(sources, codes, asts_vocabs, nls,
                                                                                sliced_flatten_ast)

        del_data_num = len(sources) % config.batch_size
        if del_data_num == 0:
            self.sources = sources
            self.codes = codes
            self.nls = nls
            self.asts_vocabs = asts_vocabs
            self.sliced_flatten_ast = sliced_flatten_ast
        else:
            self.sources = sources[:-del_data_num]
            self.codes = codes[:-del_data_num]
            self.nls = nls[:-del_data_num]
            self.asts_vocabs = asts_vocabs[:-del_data_num]
            self.sliced_flatten_ast = sliced_flatten_ast[:-del_data_num]

        a = 1

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        return self.sources[index], self.codes[index], self.asts_vocabs[index], self.nls[index], self.sliced_flatten_ast[index]

    def get_dataset(self):
        return self.sources, self.codes, self.asts_vocabs, self.nls, self.sliced_flatten_ast
