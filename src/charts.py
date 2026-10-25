"""Standalone figures built from the saved, measured experiment results."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from experiment import FAMILIES, MODEL_NAMES
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
FIGURES=ROOT/'results/figures'
FIGURES.mkdir(exist_ok=True)

def make_figures(metrics,inventory,predictions,per_family,observed,best_model):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelcolor':'#21383c','text.color':'#21383c',
        'xtick.color':'#52646a','ytick.color':'#52646a','savefig.facecolor':'#f8f8f3','figure.facecolor':'#f8f8f3'})
    labels={'Seasonal naive':'Seasonal naive','Gradient boosting':'Gradient boosting',
        'TimesFM univariate':'TimesFM · univariate','TimesFM multivariate':'TimesFM · multivariate',
        'TimesFM + calendar':'TimesFM + calendar','TimesFM + all inputs':'TimesFM + all inputs'}
    fig,axes=plt.subplots(1,2,figsize=(13.2,5.6),gridspec_kw={'wspace':.65})
    for ax,frame,value,title,fmt in [(axes[0],metrics,'wape','Forecast error · lower is better',lambda x:f'{100*x:.2f}%'),
        (axes[1],inventory,'total_cost','Simulated inventory cost · lower is better',lambda x:f'{x:,.0f}')]:
        frame=frame.sort_values(value,ascending=False)
        colors=['#07816b' if i==len(frame)-1 else '#a9b9b1' for i in range(len(frame))]
        ax.barh([labels[x] for x in frame.model],frame[value],color=colors,height=.6)
        for i,v in enumerate(frame[value]):ax.text(v+frame[value].max()*.025,i,fmt(v),va='center',fontsize=10)
        ax.set_xlim(0,frame[value].max()*1.28)
        ax.set_title(title,loc='left',fontweight='bold',pad=18)
        ax.set_facecolor('#f8f8f3')
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(left=False,bottom=False,labelbottom=False)
    fig.suptitle('Retail Demand Lab | TimesFM 3.0',x=.05,y=.98,ha='left',fontsize=19,fontweight='bold')
    fig.text(.05,.89,'One Quito store · four product families · 12 held-out weekly origins · official local inference',fontsize=11,color='#52646a')
    fig.text(.05,.04,'Forecast windows overlap. Inventory costs are simulated illustrative units, not measured retailer savings.',fontsize=10,color='#52646a')
    fig.subplots_adjust(left=.2,right=.95,top=.78,bottom=.13)
    fig.savefig(FIGURES/'results_overview.png',dpi=180,bbox_inches='tight')
    plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,family in zip(axes.flat,FAMILIES):
        frame=per_family[per_family.family==family].sort_values('wape',ascending=False)
        ax.barh([labels[x] for x in frame.model],frame.wape*100,color=['#087b67' if m==best_model else '#a9b9b1' for m in frame.model])
        ax.set_title(family,loc='left',fontweight='bold')
        ax.set_xlabel('WAPE (%)')
        ax.set_facecolor('#f8f8f3')
    fig.suptitle('Forecast accuracy differs by product family',x=.04,ha='left',fontsize=17,fontweight='bold')
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(FIGURES/'family_errors.png',dpi=170,bbox_inches='tight')
    plt.close(fig)

    daily=pd.read_csv(RESULTS/'inventory_daily.csv',parse_dates=['date'])
    fig,ax=plt.subplots(figsize=(12,4.6))
    colors=['#8d9b97','#b2b6b2','#47688e','#087b67','#bc793f','#a078a8']
    for model,color in zip(MODEL_NAMES,colors):
        g=daily[daily.model==model].groupby('date').ending_inventory.sum()
        ax.plot(g.index,g.values,label=labels[model],color=color,linewidth=1.6,alpha=.9)
    ax.set_title('Same starting inventory, different forecast-driven orders',loc='left',fontsize=16,fontweight='bold')
    ax.set_ylabel('Total ending inventory · source sales units')
    ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.18),frameon=False,fontsize=9)
    ax.set_facecolor('#f8f8f3')
    fig.autofmt_xdate()
    fig.savefig(FIGURES/'inventory_trajectory.png',dpi=170,bbox_inches='tight')
    plt.close(fig)

    origin='2017-08-02'
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,family in zip(axes.flat,FAMILIES):
        g=predictions[(predictions.model==best_model)&(predictions.family==family)&(predictions.origin==origin)]
        dates=pd.to_datetime(g.date)
        ax.fill_between(dates,g.q10,g.q90,color='#c4e4d8',alpha=.8,label='10th–90th percentile')
        ax.plot(dates,g.actual,color='#172b30',linewidth=1.8,label='Observed sales')
        ax.plot(dates,g.prediction,color='#087b67',linewidth=1.8,label='Forecast median')
        ax.set_title(family,loc='left',fontweight='bold')
        ax.tick_params(axis='x',labelrotation=25)
        ax.set_facecolor('#f8f8f3')
    handles,labels_=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels_,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('TimesFM + calendar | Forecast starts August 2, 2017',x=.04,ha='left',fontsize=17,fontweight='bold')
    fig.tight_layout(rect=(0,.06,1,.94))
    fig.savefig(FIGURES/'held_out_forecast.png',dpi=170,bbox_inches='tight')
    plt.close(fig)

