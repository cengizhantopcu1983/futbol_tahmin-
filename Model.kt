package com.futboltahmin

import org.json.JSONArray
import org.json.JSONObject

data class Skor(val skor: String, val p: Double)

data class Piyasa(val ev: Double, val beraberlik: Double, val deplasman: Double)

data class Mac(
    val lig: String,
    val ligKodu: String,
    val tarih: String?,
    val saat: String?,
    val ev: String,
    val deplasman: String,
    val pEv: Double,
    val pBeraberlik: Double,
    val pDeplasman: Double,
    val xgEv: Double,
    val xgDeplasman: Double,
    val ust25: Double,
    val kgVar: Double,
    val olasiSkorlar: List<Skor>,
    val formEv: String,
    val formDeplasman: String,
    val eloEv: Double?,
    val eloDeplasman: Double?,
    val piyasa: Piyasa?,
    val skorMatrisi: List<List<Double>>,
    val azVeri: List<String>,
) {
    val id: String get() = "$ligKodu|$tarih|$ev|$deplasman"
}

data class Tahminler(val olusturulma: String, val maclar: List<Mac>)

// ---- JSON okuma (Android'in kendi org.json kütüphanesi, ek bağımlılık yok) ----

private fun JSONObject.metin(k: String): String? = if (isNull(k)) null else getString(k)

private fun JSONObject.sayi(k: String): Double? = if (isNull(k)) null else getDouble(k)

private fun <T> JSONArray?.liste(donustur: (JSONArray, Int) -> T): List<T> =
    if (this == null) emptyList() else (0 until length()).map { donustur(this, it) }

private fun JSONObject.toMac(): Mac {
    val piyasaJson = optJSONObject("piyasa")
    return Mac(
        lig = getString("lig"),
        ligKodu = getString("lig_kodu"),
        tarih = metin("tarih"),
        saat = metin("saat"),
        ev = getString("ev"),
        deplasman = getString("deplasman"),
        pEv = getDouble("p_ev"),
        pBeraberlik = getDouble("p_beraberlik"),
        pDeplasman = getDouble("p_deplasman"),
        xgEv = getDouble("xg_ev"),
        xgDeplasman = getDouble("xg_deplasman"),
        ust25 = getDouble("ust_2_5"),
        kgVar = getDouble("kg_var"),
        olasiSkorlar = optJSONArray("olasi_skorlar").liste { a, i ->
            val o = a.getJSONObject(i)
            Skor(o.getString("skor"), o.getDouble("p"))
        },
        formEv = metin("form_ev") ?: "",
        formDeplasman = metin("form_deplasman") ?: "",
        eloEv = sayi("elo_ev"),
        eloDeplasman = sayi("elo_deplasman"),
        piyasa = piyasaJson?.let {
            Piyasa(it.getDouble("ev"), it.getDouble("beraberlik"), it.getDouble("deplasman"))
        },
        skorMatrisi = optJSONArray("skor_matrisi").liste { a, i ->
            a.getJSONArray(i).liste { satir, j -> satir.getDouble(j) }
        },
        azVeri = optJSONArray("az_veri").liste { a, i -> a.getString(i) },
    )
}

fun tahminleriOku(metin: String): Tahminler {
    val kok = JSONObject(metin)
    val maclar = kok.getJSONArray("maclar").liste { a, i -> a.getJSONObject(i).toMac() }
    return Tahminler(kok.optString("olusturulma", ""), maclar)
}
