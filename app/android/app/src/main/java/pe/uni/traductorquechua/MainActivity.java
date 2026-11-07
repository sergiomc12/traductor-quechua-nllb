package pe.uni.traductorquechua;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.speech.RecognizerIntent;
import android.speech.tts.TextToSpeech;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Locale;

/** Traductor es<->qu (Ayacucho): texto o voz de entrada, texto y voz de salida. Llama a POST /traducir. */
public class MainActivity extends Activity implements TextToSpeech.OnInitListener {
    private static final int PEDIDO_VOZ = 71;
    private static final String VACIO = "La traducción aparecerá aquí";
    private boolean esAQu = true;
    private EditText entrada;
    private TextView salida, chipOrigen, chipDestino, etiquetaEntrada, etiquetaSalida, aviso;
    private Button btnTraducir;
    private Switch autoVoz;
    private TextToSpeech tts;
    private boolean ttsListo = false;
    private String ultimaSalida = "";

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        setContentView(R.layout.activity_main);
        entrada = findViewById(R.id.entrada);
        salida = findViewById(R.id.salida);
        chipOrigen = findViewById(R.id.chipOrigen);
        chipDestino = findViewById(R.id.chipDestino);
        etiquetaEntrada = findViewById(R.id.etiquetaEntrada);
        etiquetaSalida = findViewById(R.id.etiquetaSalida);
        aviso = findViewById(R.id.aviso);
        btnTraducir = findViewById(R.id.btnTraducir);
        autoVoz = findViewById(R.id.autoVoz);
        tts = new TextToSpeech(this, this);

        findViewById(R.id.btnIntercambiar).setOnClickListener(v -> intercambiar());
        btnTraducir.setOnClickListener(v -> traducir());
        findViewById(R.id.btnMicrofono).setOnClickListener(v -> escuchar());
        findViewById(R.id.btnVozEntrada).setOnClickListener(v -> hablar(entrada.getText().toString(), esAQu ? "es" : "qu"));
        findViewById(R.id.btnVozSalida).setOnClickListener(v -> hablar(ultimaSalida, esAQu ? "qu" : "es"));
        findViewById(R.id.btnBorrar).setOnClickListener(v -> { entrada.setText(""); salida.setText(VACIO); ultimaSalida = ""; });
        findViewById(R.id.btnCopiar).setOnClickListener(v -> copiar());
        actualizarDireccion();
    }

    @Override public void onInit(int estado) { ttsListo = estado == TextToSpeech.SUCCESS; }

    @Override protected void onDestroy() {
        if (tts != null) { tts.stop(); tts.shutdown(); }
        super.onDestroy();
    }

    private void intercambiar() {
        esAQu = !esAQu;
        // La traduccion pasa a ser el nuevo texto de entrada, util para ida y vuelta
        if (!ultimaSalida.isEmpty()) entrada.setText(ultimaSalida);
        ultimaSalida = "";
        salida.setText(VACIO);
        actualizarDireccion();
    }

    private void actualizarDireccion() {
        String o = esAQu ? "Español" : "Quechua", d = esAQu ? "Quechua" : "Español";
        chipOrigen.setText(o);
        chipDestino.setText(d);
        etiquetaEntrada.setText(o.toUpperCase(Locale.ROOT));
        etiquetaSalida.setText(d.toUpperCase(Locale.ROOT));
        entrada.setHint(esAQu ? "Escribe o habla en español" : "Qillqay icha rimay runasimipi");
        aviso.setText(esAQu ? "La voz en quechua usa pronunciación aproximada con voz en español."
                : "El dictado en quechua es aproximado: el teléfono lo reconoce como español.");
    }

    private void traducir() {
        final String texto = entrada.getText().toString().trim();
        if (texto.isEmpty()) { salida.setText("Escribe o dicta un texto."); return; }
        final String origen = esAQu ? "es" : "qu", destino = esAQu ? "qu" : "es";
        btnTraducir.setEnabled(false);
        salida.setText("Traduciendo…");
        new Thread(() -> {
            String res; boolean ok = true;
            try { res = llamar(texto, origen, destino); }
            catch (Exception e) { res = "Error: " + e.getMessage(); ok = false; }
            final String fin = res; final boolean bien = ok;
            runOnUiThread(() -> {
                salida.setText(fin);
                btnTraducir.setEnabled(true);
                if (bien) {
                    ultimaSalida = fin;
                    if (autoVoz.isChecked()) hablar(fin, destino);
                }
            });
        }).start();
    }

    private void escuchar() {
        Intent i = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        i.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        i.putExtra(RecognizerIntent.EXTRA_LANGUAGE, "es-PE");  // Android no reconoce quechua
        i.putExtra(RecognizerIntent.EXTRA_PROMPT, esAQu ? "Habla en español" : "Rimay (se reconoce como español)");
        try { startActivityForResult(i, PEDIDO_VOZ); }
        catch (ActivityNotFoundException e) { Toast.makeText(this, "Este teléfono no tiene reconocimiento de voz.", Toast.LENGTH_LONG).show(); }
    }

    @Override protected void onActivityResult(int req, int res, Intent datos) {
        super.onActivityResult(req, res, datos);
        if (req != PEDIDO_VOZ || res != RESULT_OK || datos == null) return;
        ArrayList<String> r = datos.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS);
        if (r == null || r.isEmpty()) return;
        entrada.setText(r.get(0));
        entrada.setSelection(entrada.getText().length());
        traducir();
    }

    private void hablar(String texto, String idioma) {
        if (texto == null || texto.trim().isEmpty()) { Toast.makeText(this, "No hay texto para leer.", Toast.LENGTH_SHORT).show(); return; }
        if (!ttsListo) { Toast.makeText(this, "La voz del teléfono aún no está lista.", Toast.LENGTH_SHORT).show(); return; }
        // Android no trae voz en quechua: se usa la voz en español (la ortografía quechua es casi fonética)
        Locale[] candidatos = {new Locale("es", "PE"), new Locale("es", "US"), new Locale("es", "MX"), new Locale("es")};
        Locale elegido = null;
        for (Locale l : candidatos) {
            if (tts.isLanguageAvailable(l) >= TextToSpeech.LANG_AVAILABLE) { elegido = l; break; }
        }
        if (elegido == null) { Toast.makeText(this, "Instala una voz en español en Ajustes > Salida de texto a voz.", Toast.LENGTH_LONG).show(); return; }
        tts.setLanguage(elegido);
        tts.setSpeechRate("qu".equals(idioma) ? 0.85f : 1.0f);
        tts.speak(texto, TextToSpeech.QUEUE_FLUSH, null, "tq");
    }

    private void copiar() {
        if (ultimaSalida.isEmpty()) return;
        ((ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE)).setPrimaryClip(ClipData.newPlainText("traduccion", ultimaSalida));
        Toast.makeText(this, "Copiado", Toast.LENGTH_SHORT).show();
    }

    private String llamar(String texto, String origen, String destino) throws Exception {
        HttpURLConnection c = (HttpURLConnection) new URL(BuildConfig.API_URL + "/traducir").openConnection();
        c.setConnectTimeout(10000);
        c.setReadTimeout(60000);
        c.setRequestMethod("POST");
        c.setRequestProperty("Content-Type", "application/json; charset=utf-8");
        c.setDoOutput(true);
        JSONObject cuerpo = new JSONObject().put("texto", texto).put("origen", origen).put("destino", destino);
        try (OutputStream os = c.getOutputStream()) { os.write(cuerpo.toString().getBytes(StandardCharsets.UTF_8)); }
        int codigo = c.getResponseCode();
        InputStream is = codigo < 400 ? c.getInputStream() : c.getErrorStream();
        String txt = new String(is.readAllBytes(), StandardCharsets.UTF_8);
        if (codigo >= 400) throw new Exception("HTTP " + codigo);
        return new JSONObject(txt).getString("traduccion");
    }
}
