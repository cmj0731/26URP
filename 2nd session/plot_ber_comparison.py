"""Export standalone scientific BER charts and an HTML result index."""

import argparse
import html
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).with_name('.runtime-deps')))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ber_experiment import SCHEMES, theory_16qam

COLORS = ('#2458a6','#e8871a','#13927d','#a54294')
MARKERS = ('o','s','D','^')


def draw(ax, data, condition, baseline=False):
    rows = [row for row in data['results'] if row['condition']==condition and not row['noiseless']]
    resolution = 1/data['measured_bits_per_point']
    for scheme_index,(scheme,color,marker) in enumerate(zip(SCHEMES,COLORS,MARKERS)):
        selected = sorted([row for row in rows if row['waveform']==scheme],key=lambda row:row['Eb_N0_db'])
        x=np.array([row['Eb_N0_db'] for row in selected]); y=np.array([row['ber_estimate'] for row in selected])
        positive=y>0
        ax.plot(x,np.where(positive,y,np.nan),color=color,marker=marker,label=scheme,lw=1.7,ms=5)
        if np.any(positive):
            lo=np.array([row['confidence_interval'][0] for row in selected]);hi=np.array([row['confidence_interval'][1] for row in selected])
            ax.vlines(x[positive],np.maximum(lo[positive],resolution/2),hi[positive],color=color,alpha=.65,lw=1)
        if np.any(~positive):
            ax.scatter(x[~positive]+(scheme_index-1.5)*.32,np.full(np.count_nonzero(~positive),resolution),marker='v',
                       s=35,facecolors='none',edgecolors=color,zorder=4)
        reference=next(row for row in data['results'] if row['condition']==condition and row['waveform']==scheme and row['noiseless'])
        if not baseline and reference['bit_errors']:
            ax.axhline(reference['ber_estimate'],color=color,ls=':',lw=1,alpha=.7)
    if baseline:
        x=np.linspace(0,30,301)
        ax.plot(x,[theory_16qam(value) for value in x],color='#222222',ls='--',lw=1.3,label='Gray 16-QAM theory')
    ax.set_yscale('log');ax.set_ylim(resolution/2,1);ax.set_xlim(-.8,30.8)
    ax.set_xticks(range(0,31,5));ax.grid(True,which='major',alpha=.25)
    ax.set_xlabel(r'Received data-symbol $E_b/N_0$ (dB)');ax.set_ylabel('BER')
    titles={'A':'A: static three paths','B-1':'B-1: max residual Doppler 1%','B-2':'B-2: max residual Doppler 5%',
            'B-3':'B-3: max residual Doppler 10%','B-4':'B-4: max residual Doppler 20%',
            'peak':'LOS baseline: peak elevation','exit':'LOS baseline: pass exit'}
    ax.set_title(titles.get(condition,condition),fontsize=12)


def save(fig, root, name):
    fig.savefig(root/(name+'.png'),dpi=180,bbox_inches='tight')
    vector=root/(name+'.svg')
    fig.savefig(vector,bbox_inches='tight',metadata={'Date':None})
    vector.write_text('\n'.join(line.rstrip() for line in vector.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


def table(data):
    parts=['<table><tr><th>조건</th><th>방식</th><th>Eb/N0</th><th>오류 수 / 측정 비트</th><th>관측 BER</th><th>95% 프레임 기준 구간</th></tr>']
    for row in data['results']:
        lo,hi=row['confidence_interval']
        point='AWGN 없음' if row['noiseless'] else str(row['Eb_N0_db'])+' dB'
        ber='오류 미관측' if row['bit_errors']==0 else f"{row['ber_estimate']:.6g}"
        parts.append(f"<tr><td>{html.escape(row['condition'])}</td><td>{html.escape(row['waveform'])}</td><td>{point}</td><td>{row['bit_errors']:,} / {row['measured_information_bits']:,}</td><td>{ber}</td><td>[{lo:.5g}, {hi:.5g}]<br><small>오류 발생 프레임 {row['error_frame_count']}/{row['frame_count']}</small></td></tr>")
    return '\n'.join(parts)+ '</table>'


def compact_comparison(data, condition):
    parts=['<table><tr><th>Eb/N0</th>'+''.join('<th>'+html.escape(s)+'</th>' for s in SCHEMES)+'</tr>']
    for point in (10,20,30,None):
        label='AWGN 없음' if point is None else str(point)+' dB'
        cells=[]
        for scheme in SCHEMES:
            row=next(r for r in data['results'] if r['condition']==condition and r['waveform']==scheme and r['Eb_N0_db']==point)
            value='오류 미관측' if row['bit_errors']==0 else f"{row['ber_estimate']:.3g}"
            cells.append(f"<td>{value}<br><small>{row['bit_errors']:,}개 오류</small></td>")
        parts.append('<tr><td>'+label+'</td>'+''.join(cells)+'</tr>')
    return ''.join(parts)+'</table>'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--multipath',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    baseline=json.loads(args.baseline.read_text(encoding='utf-8'))
    if not baseline['complete'] or baseline['diagnostic_only']:
        raise ValueError('Only completed full-quota runs can produce final charts.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'ber-four-waveform'})
    note='Open triangles: no errors observed (at 1/N, not confidence limits; x positions offset for visibility).\nBars: pointwise 95% frame-cluster intervals. Conservative zero-error limits are in the result table.'
    doppler_note='\nDoppler percentages: maximum |residual Doppler| / subcarrier spacing (234.375 kHz).'
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,condition in zip(axes,('peak','exit')):draw(ax,baseline,condition,True)
    axes[0].legend(fontsize=8);fig.suptitle('Unit-gain LOS + AWGN, exact common-Doppler correction',fontsize=14)
    fig.text(.02,.01,note,fontsize=8);fig.tight_layout(rect=(0,.1,1,.94));save(fig,args.output,'baseline')
    content=['<h1>OFDM · AFDM · OTFS · DD-a-OFDM BER 비교</h1>',
             '<p>각 방식·조건·Eb/N0에서 <strong>10,485,760 정보 비트</strong>를 측정했다. 데이터 심볼 에너지 Es=1, 비부호화 Gray 16-QAM이며, Eb/N0는 수신 데이터 심볼 에너지 기준이다.</p>',
             '<p>신뢰구간은 독립적으로 다시 생성한 프레임을 기준으로 계산한 점별 95% 구간이다. 양의 오류 수에는 근사 군집 표준오차를, 오류 0개에는 완전한 프레임의 오류 사건에 대한 보수적 상한을 쓴다. 프레임 내부 비트 독립을 가정하지 않아 오류 미관측 구간의 상한은 넓을 수 있다. 오류가 관측되지 않았다는 사실이 정확한 BER 0이나 방식 간 동등성을 증명하지는 않는다.</p>',
             '<p>그래프의 빈 아래 삼각형은 오류 미관측을 1/N 위치에 표시한 것으로, BER 추정치나 신뢰 상한이 아니다. 겹침을 피하려고 이 삼각형만 가로 위치를 조금 벌렸다. 양의 BER에는 신뢰구간 막대를 표시하며, 실제 구간과 오류 발생 프레임 수는 아래 표에 있다.</p>',
             '<h2>단일 LOS 기준 검증</h2><p>최고점·이탈 시점의 공통 도플러를 완전히 보정했다. CP·파일럿 없는 단위 이득 채널에서 네 파형이 같은 16-QAM AWGN 이론 기준을 따르는지 확인하는 실행이다.</p><img src="baseline.svg" alt="단일 LOS BER 기준 검증">',
             '<details><summary>단일 LOS 전체 수치와 신뢰구간</summary>',table(baseline),'</details>']
    if args.multipath:
        multi=json.loads(args.multipath.read_text(encoding='utf-8'))
        if not multi['complete'] or multi['diagnostic_only']:raise ValueError('Multipath run is incomplete.')
        conditions=('A','B-1','B-2','B-3','B-4')
        fig,axes=plt.subplots(2,3,figsize=(15,9))
        for ax,condition in zip(axes.flat,conditions):draw(ax,multi,condition)
        axes.flat[5].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels()
        axes.flat[5].legend(handles,labels,loc='upper left',frameon=False)
        axes.flat[5].text(0,.48,'Matched pilot/resource budgets\nCP=5 samples per slot\nKnown P=3 and common design bounds\nEstimated channel + measured noise\nNative MMSE receivers included\nDotted lines: no-AWGN references',fontsize=10,va='top')
        fig.suptitle('Three paths: controlled residual-Doppler sweep',fontsize=15)
        fig.text(.02,.01,note+doppler_note,fontsize=8);fig.tight_layout(rect=(0,.09,1,.95));save(fig,args.output,'multipath_overview')
        for condition in conditions:
            fig,ax=plt.subplots(figsize=(7,5));draw(ax,multi,condition);ax.legend(fontsize=9)
            fig.text(.02,.01,note+doppler_note,fontsize=7);fig.tight_layout(rect=(0,.13,1,1));save(fig,args.output,'multipath_'+condition)
        content.extend(['<h2>3경로 비교</h2><p>CP 5샘플, 파일럿·보호·예약칸 25%, 총 파일럿 에너지 4,096 Es를 맞췄다. 각 방식의 파일럿 기반 채널 추정기와 MMSE 수신기를 포함하므로, 결과는 이 송수신 설계들의 비교다. 수신기에는 실제 경로 지연·도플러·이득과 잡음 전력을 제공하지 않았다.</p>',
                        '<p>지연은 0·2·5샘플, 상대 전력은 0·−10·−15 dB를 합계 1로 정규화했다. 공통 도플러 보정 후 A는 [0,0,0] Hz, B는 [0,+d,−d] Hz이며 d=2.34375·11.71875·23.4375·46.875 kHz로 증가한다. B의 반사 도플러는 통제 실험값이며 실제 스타링크 다중경로 측정값이 아니다.</p>',
                        '<img src="multipath_overview.svg" alt="3경로 잔여 도플러별 BER 비교">',
                        '<h3>B-4: 가장 큰 잔여 도플러 조건</h3>',compact_comparison(multi,'B-4'),
                        '<p>점선 수평선은 오류가 관측된 AWGN 없는 참조다. 잡음을 제거해도 채널 추정 오차는 남고 MMSE 정규화 항은 0이 되므로, 이 참조를 모든 유한 잡음 결과의 하한으로 해석하지 않는다. 초기 OFDM BEM-LS의 알려진 추정 제약도 결과에 포함되며, 곡선을 본 뒤 추정기를 다시 조정하지 않았다.</p>',
                        '<details><summary>3경로 전체 수치와 신뢰구간</summary>',table(multi),'</details>'])
    style='body{font:15px system-ui;max-width:1400px;margin:36px auto;padding:0 20px;color:#202733}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #dce2eb;text-align:left}th{background:#eef2f7;position:sticky;top:0}h2{margin-top:40px}p{max-width:1100px;line-height:1.7}small{color:#576475}summary{cursor:pointer;font-weight:600;padding:16px 0}details{overflow-x:auto}'
    (args.output/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>네 방식 BER 비교</title><style>'+style+'</style><body>'+''.join(content)+'</body></html>',encoding='utf-8')
    print('Charts and standalone result index saved:',args.output)


if __name__=='__main__':main()
