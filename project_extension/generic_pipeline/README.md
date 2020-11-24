# Some Example Executions :

python attack_launcher.py --attack LBFGSAttack --modelfile ./densenet_tl/modelfile.pth  --num_folds 1 --batch_size 50 --outdir small_batch/densenet_BFGS


python attack_launcher.py --attack LBFGSAttack --modelfile ./mobilenet_tl/modelfile.pth  --num_folds 1 --batch_size 50 --outdir small_batch/mobilenet_BFGS

python attack_launcher.py --attack LBFGSAttack --modelfile ./squeezenet_tl/modelfile.pth  --num_folds 1 --batch_size 50 --outdir small_batch/squeezenet_BFGS