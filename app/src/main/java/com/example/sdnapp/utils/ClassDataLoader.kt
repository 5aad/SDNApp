// ClassDataLoader.kt
package com.example.sdnapp.utils

import android.content.Context
import android.graphics.Color

/**
 * Loads class colors and names from CSV asset.
 */
object ClassDataLoader {
    fun loadClassData(filename: String = "class_colors.csv", context: Context): Pair<Map<Int, Int>, Map<Int, String>> {
        val colors = mutableMapOf<Int, Int>()
        val names = mutableMapOf<Int, String>()
        context.assets.open(filename).bufferedReader().useLines { lines ->
            lines.drop(1)
                .map { it.trim() }
                .filter { it.isNotEmpty() }
                .forEach { line ->
                    val parts = line.split(",")
                    val label = parts[0].toInt()
                    val r = parts[1].toInt()
                    val g = parts[2].toInt()
                    val b = parts[3].toInt()
                    val clsName = parts[4]
                    colors[label] = Color.rgb(r, g, b)
                    names[label] = clsName
                }
        }
        return colors to names
    }
}
