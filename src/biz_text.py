"""종목 사업보고서(최근 사업·반기·분기보고서) '사업의 내용' 원문을 받아 data/dart/biz/<코드>.txt 로 보관(선정 종목 설명용)."""
import os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import dart_fin as D
import io, time, urllib.request, zipfile


def text_of(rcept):
    key = os.environ['DART_KEY']
    for i in range(3):
        try:
            b = urllib.request.urlopen(f"https://opendart.fss.or.kr/api/document.xml?crtfc_key={key}&rcept_no={rcept}", timeout=180).read()
            z = zipfile.ZipFile(io.BytesIO(b))
            x = z.read(sorted(z.namelist(), key=len)[0]).decode('utf-8', 'ignore')
            t = re.sub(r'<[^>]+>', ' ', x); t = re.sub(r'\s+', ' ', t)
            # 목차가 아닌 본문의 '사업의 개요': 바로 뒤 600자 안에 '당사'·'회사'가 나오는 첫 위치
            for m in re.finditer('사업의 개요', t):
                seg = t[m.end():m.end() + 600]
                if re.search('당사|회사는|동사', seg) and '주요 제품 및 서비스' not in seg[:80]:
                    return t[m.end():m.end() + 20000]
            m = t.find('사업의 내용')
            return t[m:m + 20000] if m >= 0 else ''
        except Exception:
            time.sleep(5 * (i + 1))
    return ''

OUT = os.path.join(D.OUT, 'biz')


def get(code, refresh=False):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, code + '.txt')
    if os.path.exists(p) and not refresh:
        return open(p, encoding='utf-8').read()
    cc = D.corp_map().get(code)
    if not cc or not os.environ.get('DART_KEY'):
        return ''
    key = os.environ['DART_KEY']
    lst = []
    for ty in ('A001', 'A002', 'A003'):
        d = D._get(f"https://opendart.fss.or.kr/api/list.json?crtfc_key={key}&corp_code={cc}&bgn_de=20250101&end_de=20261231&pblntf_detail_ty={ty}&page_count=5")
        lst += [x for x in (d.get('list') or []) if '기재정정' not in x.get('report_nm', '') or True]
    if not lst:
        return ''
    rc = max(lst, key=lambda x: x['rcept_no'])
    t = text_of(rc['rcept_no'])[:20000]
    head = f"{rc['report_nm']}|{rc['rcept_no']}\n"
    open(p, 'w', encoding='utf-8').write(head + t)
    return head + t


def summary(code, n=2, cap=240):
    t = get(code)
    if not t:
        return '', ''
    head, body = t.split('\n', 1)
    sents = re.split(r'(?<=[다음])\.\s+|(?<=니다)\.\s*', body)
    cands = []
    for s in sents:
        s = s.strip()
        if not re.search(r'당사|회사는|동사', s):
            continue
        if not re.search(r'영위|제조|생산|공급|개발|판매|사업|서비스|제품', s):
            continue
        if re.search(r'참조|참고하|기재|다음의? 표|다음과 같|공급받|원재료|원료|감사|회계|연결대상|종속|한국표준산업분류|\.jpg|출처', s):
            continue
        if re.search(r'생산실적|가동|교대|설비|주식|배당|임원|해당사항|당사항|위험|변동될|최대주주|자료가 없|영업수익은|상표권|하락|감소', s):
            continue
        k = s.find('당사') if '당사' in s else max(s.find('회사는'), 0)
        if k > 40:
            continue
        s = s[k:]
        if 15 < len(s) < 260:
            cands.append(s.rstrip('.') + '.')
    # 무엇을 하는 회사인지 드러나는 문장(영위·주요 사업·전문·주력)을 먼저
    key = [c for c in cands if re.search(r'영위|주요 사업|주력|전문|주요 제품|사업부문', c)]
    pick = (key + [c for c in cands if c not in key])[:n]
    out = ' '.join(pick)
    return (out[:cap] + '…' if len(out) > cap else out), head.split('|')[0]


if __name__ == '__main__':
    for c in sys.argv[1:]:
        print(c, summary(c))
