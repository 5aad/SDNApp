package com.example.sdnapp

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Location
import android.os.Bundle
import android.os.Looper
import android.speech.RecognizerIntent
import android.speech.tts.TextToSpeech
import android.text.Html
import android.util.Log
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.sdnapp.segmentation.SegmentationProcessor
import com.google.android.gms.location.*
import com.google.android.gms.maps.CameraUpdateFactory
import com.google.android.gms.maps.GoogleMap
import com.google.android.gms.maps.MapView
import com.google.android.gms.maps.OnMapReadyCallback
import com.google.android.gms.maps.model.*
import com.google.android.libraries.places.api.Places
import com.google.android.libraries.places.api.model.Place
import com.google.android.libraries.places.api.net.FetchPlaceRequest
import com.google.android.libraries.places.api.net.FindAutocompletePredictionsRequest
import com.google.android.libraries.places.api.net.PlacesClient
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.util.Locale

/** Data class for one navigation step. */
private data class NavStep(val instruction: String, val latLng: LatLng)

class MainActivity : AppCompatActivity(),
    OnMapReadyCallback,
    TextToSpeech.OnInitListener {

    private lateinit var segmenter: SegmentationProcessor
    private lateinit var segmentationContainer: FrameLayout
    private lateinit var mapView: MapView
    private lateinit var googleMap: GoogleMap
    private lateinit var placesClient: PlacesClient
    private lateinit var tts: TextToSpeech
    private var destLatLng: LatLng? = null

    private val httpClient = OkHttpClient()

    /** Queue of navigation steps. */
    private val navSteps = mutableListOf<NavStep>()

    private lateinit var locationRequest: LocationRequest
    private lateinit var locationCallback: LocationCallback

    /** ▶️ NEW — keep handles so we can erase old graphics. */
    private var currentRoutePolyline: Polyline? = null
    private var currentDestMarker: Marker? = null  // optional

    /* -------------------------------------------------- *
     *  Speech-recognition launcher
     * -------------------------------------------------- */
    private val voiceLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            val spoken = result.data
                ?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
                ?.firstOrNull()
            if (!spoken.isNullOrBlank()) {
                tts.speak("You said $spoken", TextToSpeech.QUEUE_FLUSH, null, "UTTER1")
                lookupPlace(spoken)
            }
        }
    }

    /* -------------------------------------------------- *
     *  Permissions
     * -------------------------------------------------- */
    private fun enableMyLocationLayer() {
        if (ContextCompat.checkSelfPermission(
                this, Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            googleMap.isMyLocationEnabled = true
        }
    }

    private val locationPermLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startLocationUpdates()
        } else {
            Toast.makeText(this, "Location permission required", Toast.LENGTH_SHORT).show()
        }
    }

    /* -------------------------------------------------- *
     *  Lifecycle
     * -------------------------------------------------- */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        mapView = findViewById(R.id.mapView)
        mapView.onCreate(savedInstanceState)
        mapView.getMapAsync(this)

        tts = TextToSpeech(this, this)

        if (!Places.isInitialized()) {
            Places.initialize(applicationContext, getString(R.string.google_maps_key))
        }
        placesClient = Places.createClient(this)

        findViewById<ImageButton>(R.id.voiceButton).setOnClickListener {
            val i = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(
                    RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                    RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
                )
                putExtra(RecognizerIntent.EXTRA_PROMPT, "Please say your destination")
            }
            voiceLauncher.launch(i)
        }

        findViewById<MaterialButton>(R.id.startButton).setOnClickListener {
            val dest = destLatLng
            if (dest == null) {
                Toast.makeText(
                    this,
                    "Speak and select a destination first",
                    Toast.LENGTH_SHORT
                ).show()
                return@setOnClickListener
            }
            fetchAndRenderRoute(dest)
        }
    }

    override fun onMapReady(map: GoogleMap) {
        googleMap = map
        enableMyLocationLayer()
    }

    /* -------------------------------------------------- *
     *  Place lookup
     * -------------------------------------------------- */
    private fun lookupPlace(query: String) {
        val token = com.google.android.libraries.places.api.model.AutocompleteSessionToken.newInstance()
        val req = FindAutocompletePredictionsRequest.builder()
            .setSessionToken(token)
            .setQuery(query)
            .build()

        placesClient.findAutocompletePredictions(req)
            .addOnSuccessListener { resp ->
                val first = resp.autocompletePredictions.firstOrNull()
                if (first != null) {
                    val fetch = FetchPlaceRequest.builder(
                        first.placeId,
                        listOf(Place.Field.NAME, Place.Field.LAT_LNG)
                    ).setSessionToken(token).build()
                    placesClient.fetchPlace(fetch)
                        .addOnSuccessListener { placeResp ->
                            destLatLng = placeResp.place.latLng
                            findViewById<TextInputEditText>(R.id.destinationEditText)
                                .setText(placeResp.place.name)
                            tts.speak(
                                "${placeResp.place.name} selected",
                                TextToSpeech.QUEUE_ADD,
                                null,
                                "UTTER2"
                            )
                        }
                        .addOnFailureListener {
                            tts.speak(
                                "Error fetching place",
                                TextToSpeech.QUEUE_ADD,
                                null,
                                "UTTER_ERR"
                            )
                        }
                } else {
                    tts.speak("No places found", TextToSpeech.QUEUE_ADD, null, "UTTER3")
                }
            }
            .addOnFailureListener {
                tts.speak("Autocomplete error", TextToSpeech.QUEUE_ADD, null, "UTTER_ERR2")
            }
    }

    /* -------------------------------------------------- *
     *  Route + navigation logic
     * -------------------------------------------------- */
    private fun fetchAndRenderRoute(destination: LatLng) {
        val originRequest = LocationServices.getFusedLocationProviderClient(this)

        // Ensure we hold only one active location callback
        stopLocationUpdates()

        if (ContextCompat.checkSelfPermission(
                this, Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            locationPermLauncher.launch(Manifest.permission.ACCESS_FINE_LOCATION)
            Toast.makeText(this, "Location permission required", Toast.LENGTH_SHORT).show()
            return
        }

        originRequest.lastLocation.addOnSuccessListener { loc: Location? ->
            if (loc == null) {
                Toast.makeText(this, "Unable to get current location", Toast.LENGTH_SHORT).show()
                return@addOnSuccessListener
            }
            val origin = LatLng(loc.latitude, loc.longitude)

            Thread {
                try {
                    /* ---------- 1) Directions API ---------- */
                    val url = directionsUrl(origin, destination)
                    val resp = httpClient
                        .newCall(Request.Builder().url(url).build())
                        .execute()
                    val json = JSONObject(resp.body?.string() ?: "")
                    val route = json.getJSONArray("routes").getJSONObject(0)

                    val poly = route
                        .getJSONObject("overview_polyline")
                        .getString("points")
                    val points = decodePolyline(poly)

                    /* ---------- 2) Parse steps ---------- */
                    val stepsJson = route
                        .getJSONArray("legs")
                        .getJSONObject(0)
                        .getJSONArray("steps")
                    navSteps.clear()
                    for (i in 0 until stepsJson.length()) {
                        val s = stepsJson.getJSONObject(i)
                        val inst = Html.fromHtml(
                            s.getString("html_instructions"),
                            Html.FROM_HTML_MODE_LEGACY
                        ).toString()
                        val startLoc = s.getJSONObject("start_location")
                        val lat = startLoc.getDouble("lat")
                        val lng = startLoc.getDouble("lng")
                        navSteps.add(NavStep(inst, LatLng(lat, lng)))
                    }

                    /* ---------- 3) Location updates ---------- */
                    locationRequest = LocationRequest.create().apply {
                        interval = 4000
                        fastestInterval = 2000
                        priority = Priority.PRIORITY_HIGH_ACCURACY
                    }
                    locationCallback = object : LocationCallback() {
                        override fun onLocationResult(result: LocationResult) {
                            val myLoc = result.lastLocation ?: return
                            if (navSteps.isNotEmpty()) {
                                val next = navSteps[0]
                                val dist = FloatArray(1)
                                Location.distanceBetween(
                                    myLoc.latitude, myLoc.longitude,
                                    next.latLng.latitude, next.latLng.longitude,
                                    dist
                                )
                                if (dist[0] < 20f) {
                                    tts.speak(next.instruction, TextToSpeech.QUEUE_ADD, null, null)
                                    navSteps.removeAt(0)
                                    if (navSteps.isEmpty()) {
                                        stopLocationUpdates()
                                        runOnUiThread {
                                            startActivity(
                                                Intent(
                                                    this@MainActivity,
                                                    SegmentationCameraActivity::class.java
                                                )
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }

                    /* ---------- 4) UI thread graphics ---------- */
                    runOnUiThread {
                        // 🔄 A.  Remove previous polyline / marker
                        currentRoutePolyline?.remove()
                        currentDestMarker?.remove()

                        // 🔄 B.  Draw new route + marker
                        currentRoutePolyline = googleMap.addPolyline(
                            PolylineOptions()
                                .addAll(points)
                                .width(12f)
                        )
                        currentDestMarker = googleMap.addMarker(
                            MarkerOptions().position(destination).title("Destination")
                        )

                        // 🔄 C.  Re-centre camera
                        val bounds = LatLngBounds.builder()
                            .include(origin)
                            .include(destination)
                            .build()
                        googleMap.animateCamera(CameraUpdateFactory.newLatLngBounds(bounds, 100))

                        // 🔄 D.  Start fresh location updates
                        startLocationUpdates()
                    }

                } catch (e: Exception) {
                    Log.e("NAV", "directions error", e)
                    runOnUiThread {
                        Toast.makeText(this, "Navigation failed", Toast.LENGTH_SHORT).show()
                    }
                }
            }.start()
        }
    }

    private fun startLocationUpdates() {
        if (!::locationRequest.isInitialized || !::locationCallback.isInitialized) return
        if (ContextCompat.checkSelfPermission(
                this, Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            locationPermLauncher.launch(Manifest.permission.ACCESS_FINE_LOCATION)
            return
        }
        LocationServices.getFusedLocationProviderClient(this)
            .requestLocationUpdates(locationRequest, locationCallback, Looper.getMainLooper())
        enableMyLocationLayer()
    }

    private fun stopLocationUpdates() {
        if (::locationCallback.isInitialized) {
            LocationServices.getFusedLocationProviderClient(this)
                .removeLocationUpdates(locationCallback)
        }
    }

    private fun directionsUrl(orig: LatLng, dest: LatLng): String {
        val o = "origin=${orig.latitude},${orig.longitude}"
        val d = "destination=${dest.latitude},${dest.longitude}"
        val k = "key=${getString(R.string.google_maps_key)}"
        return "https://maps.googleapis.com/maps/api/directions/json?$o&$d&$k"
    }

    /* -------------------------------------------------- *
     *  Polyline decoder (unchanged)
     * -------------------------------------------------- */
    private fun decodePolyline(encoded: String): List<LatLng> {
        val poly = ArrayList<LatLng>()
        var index = 0
        val len = encoded.length
        var lat = 0
        var lng = 0
        while (index < len) {
            var b: Int
            var shift = 0
            var result = 0
            do {
                b = encoded[index++].code - 63
                result = result or ((b and 0x1f) shl shift)
                shift += 5
            } while (b >= 0x20)
            val dlat = if (result and 1 != 0) (result shr 1).inv() else result shr 1
            lat += dlat

            shift = 0
            result = 0
            do {
                b = encoded[index++].code - 63
                result = result or ((b and 0x1f) shl shift)
                shift += 5
            } while (b >= 0x20)
            val dlng = if (result and 1 != 0) (result shr 1).inv() else result shr 1
            lng += dlng

            poly.add(LatLng(lat.toDouble() / 1E5, lng.toDouble() / 1E5))
        }
        return poly
    }

    /* -------------------------------------------------- *
     *  MapView + TTS lifecycle
     * -------------------------------------------------- */
    override fun onResume() { super.onResume(); mapView.onResume() }
    override fun onPause()  { mapView.onPause(); super.onPause() }
    override fun onDestroy() {
        tts.shutdown()
        mapView.onDestroy()
        super.onDestroy()
    }
    override fun onLowMemory() { super.onLowMemory(); mapView.onLowMemory() }
    override fun onSaveInstanceState(out: Bundle) {
        super.onSaveInstanceState(out); mapView.onSaveInstanceState(out)
    }
    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) tts.language = Locale.getDefault()
    }
}
