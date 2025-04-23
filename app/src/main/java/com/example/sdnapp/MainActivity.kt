package com.example.sdnapp

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Location
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.google.android.gms.location.LocationServices
import com.google.android.gms.maps.model.LatLng
import com.google.android.libraries.places.api.Places
import com.google.android.libraries.places.api.model.Place
import com.google.android.libraries.places.api.net.PlacesClient
import com.google.android.libraries.places.widget.Autocomplete
import com.google.android.libraries.places.widget.model.AutocompleteActivityMode
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import androidx.core.net.toUri

class MainActivity : ComponentActivity() {

    private lateinit var placesClient: PlacesClient
    private var destLatLng: LatLng? = null

    // Fused Location Provider
    private val fusedLocationClient by lazy {
        LocationServices.getFusedLocationProviderClient(this)
    }

    // 1) Launcher for the Autocomplete Intent
    private val autocompleteLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            when (result.resultCode) {
                Activity.RESULT_OK -> {
                    val place = Autocomplete.getPlaceFromIntent(result.data!!)
                    destLatLng = place.location
                    findViewById<TextInputEditText>(R.id.destinationEditText)
                        .setText(place.displayName)
                }
                else -> {
                    val status = Autocomplete.getStatusFromIntent(result.data!!)
                    Toast.makeText(this, "Error: ${status.statusMessage}", Toast.LENGTH_SHORT).show()
                }
            }
        }

    // 2) Permission launcher for location
    private val requestLocationPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (!granted) {
                Toast.makeText(this, "Location permission required", Toast.LENGTH_SHORT).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // --- Initialize Places SDK ---
        if (!Places.isInitialized()) {
            Places.initialize(applicationContext, getString(R.string.google_maps_key))
        }
        placesClient = Places.createClient(this)

        // --- Wire up destination field to launch Autocomplete ---
        findViewById<TextInputEditText>(R.id.destinationEditText)
            .setOnClickListener {
                val fields = listOf(Place.Field.ID, Place.Field.NAME, Place.Field.LAT_LNG)
                val intent = Autocomplete.IntentBuilder(AutocompleteActivityMode.FULLSCREEN, fields)
                    .build(this)
                autocompleteLauncher.launch(intent)
            }

        // --- Ensure location permission ---
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestLocationPermission.launch(Manifest.permission.ACCESS_FINE_LOCATION)
        }

        // --- Handle Start button click ---
        findViewById<MaterialButton>(R.id.startButton).setOnClickListener {
            val dest = destLatLng
            if (dest == null) {
                Toast.makeText(this, "Please select a destination first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            fusedLocationClient.lastLocation
                .addOnSuccessListener { location: Location? ->
                    val navUri =
                        "google.navigation:q=${dest.latitude},${dest.longitude}&mode=w".toUri()
                    val navIntent = Intent(Intent.ACTION_VIEW, navUri).apply {
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    }
                    val intent = Intent(this, SegmentationCameraActivity::class.java)
                    startActivity(intent)
                    startActivity(navIntent)

                }
                .addOnFailureListener { e ->
                    Toast.makeText(this, "Location error: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
                }
        }
    }
}
