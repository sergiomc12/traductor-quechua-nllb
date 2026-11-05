package pe.uni.traductorquechua;

import android.app.Activity;
import android.os.Bundle;
import android.text.InputType;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

/** Cliente minimo del traductor es<->qu (Ayacucho): una pantalla, llama a POST /traducir. */
public class MainActivity extends Activity {
    private boolean esAQu = true;
    private Button btnDir;
    private EditText entrada;
    private TextView salida;
    private Button btnTraducir;

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        int p = (int) (16 * getResources().getDisplayMetrics().density);
        LinearLayout raiz = new LinearLayout(this);
        raiz.setOrientation(LinearLayout.VERTICAL);
        raiz.setPadding(p, p, p, p);

        btnDir = new Button(this);
        btnDir.setOnClickListener(v -> { esAQu = !esAQu; actualizarDireccion(); });
        entrada = new EditText(this);
        entrada.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        entrada.setMinLines(4);
        entrada.setGravity(Gravity.TOP);
        btnTraducir = new Button(this);
        btnTraducir.setText("Traducir");
        btnTraducir.setOnClickListener(v -> traducir());
        salida = new TextView(this);
        salida.setTextSize(18);
        salida.setTextIsSelectable(true);
        salida.setPadding(0, p, 0, 0);

        raiz.addView(btnDir, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        raiz.addView(entrada, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        raiz.addView(btnTraducir, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        raiz.addView(salida);
        setContentView(raiz);
        actualizarDireccion();
    }

    private void actualizarDireccion() {
        btnDir.setText(esAQu ? "Español → Quechua (Ayacucho)  ⇄" : "Quechua (Ayacucho) → Español  ⇄");
        entrada.setHint(esAQu ? "Escribe en español" : "Qillqay runasimipi");
    }

    private void traducir() {
        final String texto = entrada.getText().toString().trim();
        if (texto.isEmpty()) { salida.setText("Escribe un texto."); return; }
        final String origen = esAQu ? "es" : "qu", destino = esAQu ? "qu" : "es";
        btnTraducir.setEnabled(false);
        salida.setText("Traduciendo…");
        new Thread(() -> {
            String res;
            try { res = llamar(texto, origen, destino); }
            catch (Exception e) { res = "Error: " + e.getMessage(); }
            final String fin = res;
            runOnUiThread(() -> { salida.setText(fin); btnTraducir.setEnabled(true); });
        }).start();
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
