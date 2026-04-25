"""Parse SAE training log output and plot loss curves."""
import re
import matplotlib.pyplot as plt

LOG = """
step=     0  epoch=0  loss=0.00013  recon=0.00013  aux=0.00000  l0=64.0  dead=0.0%  t=0s
step=    50  epoch=0  loss=0.00012  recon=0.00012  aux=0.00000  l0=64.0  dead=0.0%  t=9s
step=   100  epoch=0  loss=0.00010  recon=0.00010  aux=0.00000  l0=64.0  dead=0.0%  t=18s
step=   150  epoch=0  loss=0.00009  recon=0.00009  aux=0.00000  l0=64.0  dead=0.0%  t=27s
step=   200  epoch=0  loss=0.00008  recon=0.00008  aux=0.00000  l0=64.0  dead=0.0%  t=36s
step=   250  epoch=1  loss=0.00008  recon=0.00007  aux=0.00009  l0=64.0  dead=11.7%  t=48s
step=   300  epoch=1  loss=0.00007  recon=0.00007  aux=0.00006  l0=64.0  dead=23.7%  t=61s
step=   350  epoch=1  loss=0.00007  recon=0.00007  aux=0.00006  l0=64.0  dead=18.6%  t=74s
step=   400  epoch=1  loss=0.00006  recon=0.00006  aux=0.00007  l0=64.0  dead=13.3%  t=87s
step=   450  epoch=2  loss=0.00006  recon=0.00006  aux=0.00007  l0=64.0  dead=8.9%  t=99s
step=   500  epoch=2  loss=0.00006  recon=0.00006  aux=0.00008  l0=64.0  dead=5.9%  t=112s
step=   550  epoch=2  loss=0.00006  recon=0.00006  aux=0.00008  l0=64.0  dead=3.5%  t=126s
step=   600  epoch=2  loss=0.00006  recon=0.00006  aux=0.00009  l0=64.0  dead=1.9%  t=139s
step=   650  epoch=2  loss=0.00006  recon=0.00006  aux=0.00009  l0=64.0  dead=0.9%  t=152s
step=   700  epoch=3  loss=0.00006  recon=0.00006  aux=0.00009  l0=64.0  dead=0.8%  t=165s
step=   750  epoch=3  loss=0.00006  recon=0.00005  aux=0.00009  l0=64.0  dead=0.7%  t=178s
step=   800  epoch=3  loss=0.00006  recon=0.00005  aux=0.00009  l0=64.0  dead=0.4%  t=190s
step=   850  epoch=3  loss=0.00006  recon=0.00005  aux=0.00010  l0=64.0  dead=0.3%  t=203s
step=   900  epoch=4  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.3%  t=216s
step=   950  epoch=4  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.3%  t=229s
step=  1000  epoch=4  loss=0.00006  recon=0.00005  aux=0.00009  l0=64.0  dead=0.3%  t=242s
step=  1050  epoch=4  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.2%  t=256s
step=  1100  epoch=4  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.1%  t=269s
step=  1150  epoch=5  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.2%  t=281s
step=  1200  epoch=5  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.2%  t=294s
step=  1250  epoch=5  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.3%  t=307s
step=  1300  epoch=5  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.1%  t=320s
step=  1350  epoch=6  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.1%  t=333s
step=  1400  epoch=6  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.2%  t=346s
step=  1450  epoch=6  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.2%  t=358s
step=  1500  epoch=6  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.1%  t=371s
step=  1550  epoch=6  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.1%  t=385s
step=  1600  epoch=7  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.1%  t=398s
step=  1650  epoch=7  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.1%  t=411s
step=  1700  epoch=7  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.1%  t=424s
step=  1750  epoch=7  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.0%  t=437s
step=  1800  epoch=8  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.0%  t=449s
step=  1850  epoch=8  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.1%  t=462s
step=  1900  epoch=8  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.1%  t=475s
step=  1950  epoch=8  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.1%  t=488s
step=  2000  epoch=8  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.0%  t=501s
step=  2050  epoch=9  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.0%  t=515s
step=  2100  epoch=9  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.1%  t=528s
step=  2150  epoch=9  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.0%  t=541s
step=  2200  epoch=9  loss=0.00005  recon=0.00005  aux=0.00009  l0=64.0  dead=0.0%  t=553s
"""

pattern = re.compile(
    r"step=\s*(\d+)\s+epoch=(\d+)\s+loss=([\d.]+)\s+recon=([\d.]+)\s+aux=([\d.]+)"
    r"\s+l0=([\d.]+)\s+dead=([\d.]+)%"
)

steps, losses, recons, auxs, deads = [], [], [], [], []
for m in pattern.finditer(LOG):
    steps.append(int(m.group(1)))
    losses.append(float(m.group(3)))
    recons.append(float(m.group(4)))
    auxs.append(float(m.group(5)))
    deads.append(float(m.group(7)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax1.plot(steps, recons, label="Reconstruction loss", color="steelblue")
ax1.plot(steps, losses, label="Total loss", color="steelblue", linestyle="--", alpha=0.5)
ax1.set_ylabel("MSE Loss")
ax1.set_title("BatchTopK SAE Training — METAGENE-1 Layer 32")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(steps, deads, color="tomato")
ax2.set_ylabel("Dead features (%)")
ax2.set_xlabel("Training step")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = "model/sae_training_curves.png"
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
