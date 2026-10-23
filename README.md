# Futbol Tahmin Modeli (Aşama 1)

Süper Lig, Premier League, La Liga ve Serie A maçları için olasılık tahmini.
Veri: football-data.co.uk (sonuçlar, bahis oranları, fikstür; ücretsiz, otomatik indirilir).

## Kurulum
```
pip install -r requirements.txt
```
Tüm dosyalar aynı klasörde (`futbol_tahmin/`) olmalı. İlk çalıştırmada veriler `futbol_tahmin/data/` klasörüne indirilir.

## Kullanım
```
python predict.py --lig super-lig --ev Galatasaray --deplasman Fener
python predict.py --fikstur                         # 4 ligin yaklaşan maçları
python predict.py --fikstur --json tahminler.json   # Android uygulaması için JSON
python predict.py --lig premier --siralama          # güç sıralaması
python predict.py --lig seriea --takimlar           # takım adları
python backtest.py --hepsi --sezon 2526             # geçen sezonda model ne kadar iyiydi?
```
Lig adları: `super-lig`, `premier`, `laliga`, `seriea`. Takım adı kısmi yazılabilir ("Fener", "man city").

## Nasıl çalışıyor
1. **Dixon-Coles modeli** her takım için hücum ve savunma gücü, ev sahibi avantajı ve
   düşük skorlu maç düzeltmesi hesaplar. Son 4 sezon kullanılır; eski maçların ağırlığı
   yaklaşık 1 yılda yarıya iner (`--xi` ile değişir). Form etkisi buradan gelir.
2. Beklenen gollere belirsizlik eklenip maç **10.000 kez simüle edilir**; 1/X/2, skor,
   2.5 üst, karşılıklı gol olasılıkları çıkar.
3. **Elo** puanı ikinci görüş olarak gösterilir.
4. Son 5 maç formu ve geçmiş karşılaşmalar bilgi amaçlı gösterilir (H2H modele girmez,
   çünkü istatistiksel olarak zayıf bir gösterge).

## Sonuçları nasıl okumalı
`backtest.py` modeli bahis piyasasıyla kıyaslar. Piyasa futboldaki en güçlü referanstır;
modelin ona yaklaşması iyi sonuçtur, düzenli olarak yenmesi beklenmemeli.
Kalibrasyon tablosu "%60 dediğimiz maçların gerçekten ~%60'ı oldu mu?" sorusunu yanıtlar.

## Yol haritası
- Aşama 2: xG verisi (şut kalitesi) ile form katmanı
- Aşama 3: ilk 11 ve sakatlık düzeltmesi (API-Football vb.)
- Aşama 4: taktik profiller (pres, topa sahip olma) ve duran top bileşeni
- Android: tahminler JSON olarak yayınlanır, uygulama bunu gösterir
