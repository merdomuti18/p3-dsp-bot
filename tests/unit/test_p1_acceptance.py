"""Independent acceptance scenarios for the defects missed by PR #7.

Expected money values use arithmetic/Decimal, never production fill helpers.
"""
import copy
import json
from datetime import datetime, timedelta
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import portfoy_yonetici as manager
import mott_state as state
import mott_aylik_rapor as report
from p1_safety import StateConflict, start_ledger, reconcile


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, 'PORTFOY_FILE', tmp_path/'portfoy.json')
    monkeypatch.setattr(manager, 'PORTFOY_AUDIT_FILE', tmp_path/'audit.jsonl')
    manager._FIYAT_CACHE.clear()


def test_stale_writer_does_not_create_10000_tl(monkeypatch):
    old={'baslangic':100000,'nakit':90000,'pozisyonlar':{'A':{'lotlar':100,'giris_f':100}},'_gen':5,'_initial_gen':5}
    disk={'baslangic':100000,'nakit':80000,'pozisyonlar':{'A':{'lotlar':100,'giris_f':100},'B':{'lotlar':200,'giris_f':50}},'_gen':6}
    manager.PORTFOY_FILE.write_text(json.dumps(disk),encoding='utf-8')
    before=manager.PORTFOY_FILE.read_bytes()
    with pytest.raises(StateConflict):
        manager.portfoy_kaydet(old)
    assert manager.PORTFOY_FILE.read_bytes()==before
    assert disk['nakit']+sum(p['lotlar']*p['giris_f'] for p in disk['pozisyonlar'].values())==100000


def test_same_generation_two_saves_only_one_commits(monkeypatch):
    manager.PORTFOY_FILE.write_text(json.dumps({'nakit':100000,'pozisyonlar':{},'_gen':0}),encoding='utf-8')
    a=manager.portfoy_yukle(); b=manager.portfoy_yukle()
    def save(p):
        try:
            manager.portfoy_kaydet(p)
            return 'pass'
        except StateConflict:
            return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as executor:
        results=list(executor.map(save,[a,b]))
    assert sorted(results)==['conflict','pass']
    assert json.loads(manager.PORTFOY_FILE.read_text())['nakit']==100000


def test_stale_writer_does_not_resurrect_closed_position():
    stale={'nakit':90000,'pozisyonlar':{'A':{'giris_f':100,'lotlar':100}},'_gen':0,'_initial_gen':0}
    disk={'nakit':100000,'pozisyonlar':{},'_gen':1}
    manager.PORTFOY_FILE.write_text(json.dumps(disk),encoding='utf-8')
    with pytest.raises(StateConflict):
        manager.portfoy_kaydet(stale)
    assert not json.loads(manager.PORTFOY_FILE.read_text())['pozisyonlar']


def test_report_uses_real_normalizer_lots_and_money(tmp_path, monkeypatch):
    raw={'nakit':100000,'pozisyonlar':{},'trade_history':[
        {'symbol':'A','giris_fiyat':100,'cikis_fiyat':110,'lotlar':100,'tl_kar':995,'position_id':'P1_A','position_closed':True,'neden':'STOP'},
        {'symbol':'B','giris_fiyat':100,'cikis_fiyat':90,'lotlar':10,'tl_kar':-101,'position_id':'P1_B','position_closed':True,'neden':'STOP'}]}
    (tmp_path/'portfoy.json').write_text(json.dumps(raw),encoding='utf-8')
    monkeypatch.setattr(state,'BASE',tmp_path); monkeypatch.setattr(report,'BASE',tmp_path)
    assert report._p1_p2_rapor_blok('P1')['kar_faktoru']==9.85


def test_one_lot_tp1_counts_as_completed_position(tmp_path, monkeypatch):
    raw={'nakit':100000,'pozisyonlar':{},'trade_history':[{'position_id':'P1_A','position_closed':True,'neden':'TP1','lotlar':1,'tl_kar':5}]}
    (tmp_path/'portfoy.json').write_text(json.dumps(raw),encoding='utf-8')
    monkeypatch.setattr(state,'BASE',tmp_path); monkeypatch.setattr(report,'BASE',tmp_path)
    result=report._p1_p2_rapor_blok('P1')
    assert result['tamamlanan_pozisyon_sayisi']==1
    assert result['kar_faktoru'] is None and result['kar_faktoru_sonsuz']


@pytest.mark.parametrize('source,time,eligible',[
    ('yfinance_1m','2026-10-06T10:45:00+03:00',True),
    ('yfinance_1m','2026-10-01T10:45:00+03:00',False),
    ('yfinance_1m','2026-10-06T12:00:00+03:00',False),
    ('yfinance_1d_close','2026-10-01T00:00:00+03:00',False),
    ('tradingview',None,False),
    ('yfinance_fast_info',None,False),
])
def test_quote_age_gate(source,time,eligible):
    q=manager._quote_quality({'price':100,'valid':True,'source':source,'time':time},datetime(2026,10,6,11))
    assert q['trade_eligible'] is eligible


def test_invalid_macro_time_cannot_trigger_emergency_liquidation():
    p={'makro_karar':{'karar':'GIRME','skor':99,'gecerlilik_tarih':'2026-10-06','zaman':'garbage'}}
    decision,score,meta=manager.get_aktif_makro_karar(p,now=datetime(2026,10,6,11))
    assert decision=='GIRME' and score==0 and meta['valid'] is False


def test_acil_exit_cash_and_net_profit_include_both_fees(monkeypatch):
    p={'nakit':89995.0,'pozisyonlar':{'A':{'giris_f':100,'lotlar':100,'entry_lotlar':100,'entry_komisyon':5,'position_id':'P1_A'}},'trade_history':[]}
    monkeypatch.setattr(manager,'saatlik_bar',lambda *a,**k:None)
    monkeypatch.setattr(manager,'guncel_fiyat_detayli',lambda *a,**k:{'price':110,'trade_eligible':True,'source':'test','time':'2026-10-06T11:00:00+03:00'})
    p,_=manager.pozisyon_guncelle_saatlik(p,'GIRME',99,now=datetime(2026,10,6,11))
    # 100*109.89 minus selling commission 5.4945, less entry cost 10005.
    assert p['nakit']==pytest.approx(100978.5055)
    assert p['trade_history'][0]['tl_kar']==pytest.approx(978.5055)
    assert reconcile(p)['status']=='pass'


def test_entry_half_exit_final_exit_net_profit_matches_cash(monkeypatch):
    p={'nakit':100000,'pozisyonlar':{}}
    monkeypatch.setattr(manager,'guncel_fiyat_detayli',lambda *a,**k:{'price':100,'valid':True,'trade_eligible':True,'source':'test','time':'2026-10-06T11:00:00+03:00'})
    monkeypatch.setattr(manager.mott_risk,'kitaplar_arasi_sayi',lambda *a,**k:0)
    p,_,_,_=manager.yeni_pozisyon_ac(p,[{'symbol':'A','final_score':80}],'NORMAL',{'size_factor':1},now=datetime(2026,10,6,11))
    original_lots=p['pozisyonlar']['A']['lotlar']
    assert original_lots==200
    bar={'open':102,'high':109,'low':101,'close':107,'volume':1000,'bar_time':'2026-10-06T11:00:00+03:00'}
    monkeypatch.setattr(manager,'saatlik_bar',lambda *a,**k:bar)
    p,_=manager.pozisyon_guncelle_saatlik(p,'NORMAL',now=datetime(2026,10,6,12))
    before=copy.deepcopy(p)
    p,_=manager.pozisyon_guncelle_saatlik(p,'NORMAL',now=datetime(2026,10,6,12))
    assert p==before
    bar.update(open=106,high=107,low=102,close=103,bar_time='2026-10-06T12:00:00+03:00')
    p,_=manager.pozisyon_guncelle_saatlik(p,'NORMAL',now=datetime(2026,10,6,13))
    expected=Decimal('100000')-Decimal('20000')-Decimal('10')
    for price in (Decimal('107.892'),Decimal('103.44645')):
        # Fill prices are rounded to four decimals by the documented simulator.
        price=price.quantize(Decimal('.0001'))
        expected += 100*price-(100*price*Decimal('.0005')).quantize(Decimal('.0001'))
    assert not p['pozisyonlar']
    assert p['nakit']==pytest.approx(float(expected),abs=.001)
    assert sum(t['tl_kar'] for t in p['trade_history'])==pytest.approx(p['nakit']-100000,abs=.001)
    assert reconcile(p)['status']=='pass'


def test_ledger_does_not_hide_legacy_gap():
    p={'nakit':94120.673962,'pozisyonlar':{},'islem_defteri':[]}
    start_ledger(p,'2026-10-06')
    assert p['muhasebe_acilis']['nakit']==94120.673962
    assert reconcile(p)['status']=='pass'
    p['nakit']+=10
    with pytest.raises(ValueError,match='reconciliation'):
        reconcile(p)


def test_liquidity_field_survives_scanner_to_candidate(monkeypatch):
    import scanner_p1
    ind={'close':100,'vol20':200000,'rsi':55,'rel_vol':1.2,'change_pct':1,'adx':25,'cmf':.1,'alpha_bull':False,'alpha_trend_bull':True}
    row=scanner_p1._build_signal_records('06.10.2026 19:00','test',{'ZKN':[{'symbol':'A','ind':ind}]})[0]
    assert row['islem_tl']==20000000
    monkeypatch.setattr(manager,'ma_motor_skoru',lambda s:None)
    monkeypatch.setattr(manager,'alpha_trend_analiz',lambda s:{'state':'none','flip_up':False,'bonus':0,'tag':''})
    result=manager.build_candidate('A',row,None,{'score':0,'label':'Normal'})
    assert result['final_score']==20  # 14 ZKN + 6 single confirmation; no false -4 penalty.


def test_zkn_kbm_boundary_and_direction():
    import scanner_p1 as scan
    ind={'close':110,'ema50':100,'ema200':95,'rsi':50,'stochrsi':30,'cmf':.1,'rel_vol':1,
         'bb_mid':105,'macd':2,'macd_sig':1,'macd_prev':.5,'macd_sprev':1}
    assert scan.strategy_zkn(ind,20000000)
    assert scan.strategy_karisik_bb_macd(ind,20000000)
    assert not scan.strategy_zkn({**ind,'close':99},20000000)
    assert not scan.strategy_karisik_bb_macd({**ind,'macd_prev':2},20000000)
    assert not scan.strategy_zkn(ind,9999999)


def test_future_and_open_daily_bars_cannot_change_agents():
    from p1_safety import closed_daily
    ref=datetime(2026,10,6,11)
    dates=pd.date_range('2026-09-28',periods=12,freq='B')
    frame=pd.DataFrame({'Close':np.arange(12)+100},index=dates)
    past=frame.loc[frame.index < pd.Timestamp('2026-10-06')]
    pd.testing.assert_frame_equal(closed_daily(frame,ref),past)
    frame.loc[frame.index>=pd.Timestamp('2026-10-06'),'Close']=999999
    pd.testing.assert_frame_equal(closed_daily(frame,ref),past)


def test_holiday_blocks_real_entry_path(monkeypatch):
    monkeypatch.setattr(manager,'guncel_fiyat_detayli',lambda *a,**kw:pytest.fail('holiday should block before price fetch'))
    p={'nakit':100000,'pozisyonlar':{}}
    result=manager.yeni_pozisyon_ac(p,[{'symbol':'A','final_score':80}],'NORMAL',{},now=datetime(2026,10,29,11))
    assert result[2]==[] and result[3][0]['reason']=='seans_kapali'


def test_stale_quote_blocks_entry_even_if_price_is_positive(monkeypatch):
    q=manager._quote_quality({'price':100,'valid':True,'source':'yfinance_1m','time':'2026-10-01T11:00:00+03:00'},datetime(2026,10,6,11))
    monkeypatch.setattr(manager,'guncel_fiyat_detayli',lambda *a,**k:q)
    p={'nakit':100000,'pozisyonlar':{}}
    p,_,bought,_=manager.yeni_pozisyon_ac(p,[{'symbol':'A','final_score':80}],'NORMAL',{'size_factor':1},now=datetime(2026,10,6,11))
    assert not bought and p['nakit']==100000 and not p['pozisyonlar']


def test_failed_atomic_replace_preserves_old_economic_state(monkeypatch):
    import mott_state_coordination as coordination
    disk={'nakit':100000,'pozisyonlar':{},'_gen':0}
    manager.PORTFOY_FILE.write_text(json.dumps(disk),encoding='utf-8')
    before=manager.PORTFOY_FILE.read_bytes()
    p=manager.portfoy_yukle()
    monkeypatch.setattr(coordination.os,'replace',lambda *a: (_ for _ in ()).throw(OSError('simulated crash before replace')))
    with pytest.raises(OSError):
        manager.portfoy_kaydet(p)
    assert manager.PORTFOY_FILE.read_bytes()==before


def test_ledger_rejects_duplicate_event_and_negative_lots():
    p={'nakit':100000,'pozisyonlar':{}}
    start_ledger(p,'test')
    p['islem_defteri']=[{'event_id':'1','symbol':'A','islem_tipi':'ALIS','lot':1,'nakit_etkisi':-100}]*2
    with pytest.raises(ValueError,match='Duplicate'):
        reconcile(p)
    p['islem_defteri']=[{'event_id':'1','symbol':'A','islem_tipi':'SATIS','lot':1,'nakit_etkisi':100}]
    with pytest.raises(ValueError,match='Negative'):
        reconcile(p)


def test_p2_trade_helper_keeps_pre_p1_contract():
    p={'trade_history':[]}
    manager._p2_trade_kaydet(p,'A',{'giris_f':100,'lotlar':100,'giris_t':'01.10.2026'},110,'STOP')
    assert 'islem_defteri' not in p
    assert 'tl_kar' not in p['trade_history'][0]
    assert p['trade_history'][0]['pnl_pct']==10


def test_unpriced_equity_is_not_reported_as_valid_return(tmp_path, monkeypatch):
    p={'nakit':90000,'pozisyonlar':{'A':{'lotlar':100,'giris_f':100}},'trade_history':[]}
    (tmp_path/'portfoy.json').write_text(json.dumps(p),encoding='utf-8')
    monkeypatch.setattr(state,'BASE',tmp_path); monkeypatch.setattr(report,'BASE',tmp_path)
    monkeypatch.setattr(manager,'guncel_fiyat_detayli',lambda *a,**kw:{'price':100,'valuation_valid':False})
    result=report._p1_p2_rapor_blok('P1')
    assert result['degerleme_eksik'] and result['getiri_pct'] is None
    daily=manager.gunluk_equity_kaydet(p,now=datetime(2026,10,6,18))['gunluk_equity_tarihcesi'][-1]
    assert daily['equity'] is None and daily['degerleme_eksik']==['A']


def test_profit_factor_aggregates_partial_sales_by_position(tmp_path, monkeypatch):
    raw={'nakit':100000,'pozisyonlar':{},'trade_history':[
        {'position_id':'P1_A','position_closed':False,'neden':'TP1','tl_kar':100},
        {'position_id':'P1_A','position_closed':True,'neden':'STOP','tl_kar':-200},
        {'position_id':'P1_B','position_closed':True,'neden':'STOP','tl_kar':500},
        {'position_id':'P1_C','position_closed':False,'neden':'TP1','tl_kar':1000}]}
    (tmp_path/'portfoy.json').write_text(json.dumps(raw),encoding='utf-8')
    monkeypatch.setattr(state,'BASE',tmp_path); monkeypatch.setattr(report,'BASE',tmp_path)
    result=report._p1_p2_rapor_blok('P1')
    assert result['kar_faktoru']==5 and result['tamamlanan_pozisyon_sayisi']==2


def test_workflow_writers_have_one_dependency_order():
    import yaml
    from pathlib import Path
    wf=yaml.safe_load((Path(__file__).parents[2]/'.github/workflows/mott_daily.yml').read_text(encoding='utf-8'))
    jobs=wf['jobs']
    order=['p3-dsp','p1-momentum','p2-smc','p3-monitor','p4-meta','p4-monitor','p5-komite','p5-monitor']
    for previous,current in zip(order,order[1:]):
        assert previous in jobs[current]['needs']
        assert 'always()' in jobs[current]['if'] and '!failure()' in jobs[current]['if']
    p1_commit=next(s['run'] for s in jobs['p1-momentum']['steps'] if s.get('name')=='State commit')
    assert 'git reset --soft' not in p1_commit
    assert 'git diff --quiet "$STATE_BASE" origin/main -- $FILES' in p1_commit
