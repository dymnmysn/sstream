import torchaudio
import torch

from soundstream import StreamableModel

if __name__=='__main__':
    mmodel = StreamableModel(
            batch_size=32,
            sample_rate=16_000,
            segment_length=32270,
            padding='same',
            dataset='librispeech')

    ckpt = '/ari/users/ibaskaya/projeler/sstream/lightning_logs/version_10/checkpoints/last.ckpt'
    infile = '/ari/users/ibaskaya/projeler/sstream/data/testdata/908_157963_908-157963-0028.in.wav'
    outfile = '/ari/users/ibaskaya/projeler/sstream/data/testdata/outputs/cikti.wav'
    mmodel.load_state_dict(torch.load(ckpt,'cpu')['state_dict'])
    _ = mmodel.eval()

    x, sr = torchaudio.load(infile)
    x, sr = torchaudio.functional.resample(x, sr, 16000), 16000
    with torch.no_grad():
        y = mmodel.encode(x.unsqueeze(0))
        # y = y[:, :, :4]  # if you want to reduce code size.
        z = mmodel.decode(y)
    torchaudio.save(outfile, z.squeeze(0), sr)