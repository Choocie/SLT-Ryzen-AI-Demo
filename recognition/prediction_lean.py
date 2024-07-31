import warnings
from google.protobuf.reflection import ParseMessage
from modelling.model_lean import build_model
warnings.filterwarnings("ignore")
import argparse
import os, sys
sys.path.append(os.getcwd())#slt dir
import torch
from utils.misc import (
    get_logger,
    load_config,
    make_logger, move_to_device,
    neq_load_customized
)
from dataset.Dataloader import build_dataloader


def evaluation(model, val_dataloader, cfg, 
        tb_writer=None, wandb_run=None,
        epoch=None, global_step=None,
        generate_cfg={}, save_dir=None,
        do_translation=True, do_recognition=True):  
    logger = get_logger()
    logger.info(generate_cfg)

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            batch = move_to_device(batch, cfg['device'])
            dummy_input = (False, batch['recognition_inputs'], batch)
            
            torch.onnx.export(model, dummy_input , "s2g_singlestream.onnx", verbose=True,opset_version=13)

            forward_output = model(is_train=False, **batch)  

            for k, gls_logits in forward_output.items():
                if gls_logits is None: continue
                if type(gls_logits) == dict:
                    for kk, vv in gls_logits.items():
                        print(k,kk, vv)
                    else:
                        print(k, gls_logits)

            if do_recognition:
                for k, gls_logits in forward_output.items():
                    if k == 'ctc_decode_output':
                        print('ctc_decode_output',gls_logits)
                    if not 'gloss_logits' in k or gls_logits==None:
                        continue
                    logits_name = k.replace('gloss_logits','')
                    if logits_name in ['rgb_','keypoint_','fuse_','ensemble_last_','ensemble_early_','']:
                        if logits_name=='ensemble_early_':
                            input_lengths = forward_output['aux_lengths']['rgb'][-1]
                        else:
                            input_lengths = forward_output['input_lengths']

                        ctc_decode_output = model.predict_gloss_from_logits(
                            gloss_logits=gls_logits, 
                            beam_size=generate_cfg['recognition']['beam_size'], 
                            input_lengths=input_lengths
                        ) 
                        print('CTC decoding output:',ctc_decode_output)
                        batch_pred_gls = model.gloss_tokenizer.convert_ids_to_tokens(ctc_decode_output)
                        print('Predicted glosses:',batch_pred_gls)
                    else:
                        print(logits_name)
                        raise ValueError
    
            if do_translation:
                generate_output = model.generate_txt(
                    transformer_inputs=forward_output['transformer_inputs'],
                    generate_cfg=generate_cfg['translation'])
                
                print('decoded_sequences',generate_output['decoded_sequences'])
    return []
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser("SLT baseline Testing")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--save_subdir",
        default='prediction',
        type=str
    )
    parser.add_argument(
        '--ckpt_name',
        default='best.ckpt',
        type=str
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_dir = cfg['training']['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    global logger
    logger = make_logger(model_dir=model_dir, log_file='prediction.log')
    device_ = 'cpu'
    cfg['device'] = torch.device(device_)
    model = build_model(cfg)
    #load model
    load_model_path = os.path.join(model_dir,'ckpts',args.ckpt_name)
    if os.path.isfile(load_model_path):
        state_dict = torch.load(load_model_path, map_location=device_)
        neq_load_customized(model, state_dict['model_state'], verbose=True)
        epoch, global_step = state_dict.get('epoch',0), state_dict.get('global_step',0)
        logger.info('Load model ckpt from '+load_model_path)
    else:
        logger.info(f'{load_model_path} does not exist')
        epoch, global_step = 0, 0
    do_translation, do_recognition = cfg['task']!='S2G', cfg['task']!='G2T'
    for split in ['test',]:
        logger.info('Evaluate on {} set'.format(split))
        dataloader, sampler = build_dataloader(cfg, split, model.text_tokenizer, model.gloss_tokenizer)
        evaluation(model=model, val_dataloader=dataloader, cfg=cfg, 
                epoch=epoch, global_step=global_step, 
                generate_cfg=cfg['testing']['cfg'],
                save_dir=os.path.join(model_dir,args.save_subdir,split),
                do_translation=do_translation, do_recognition=do_recognition)

