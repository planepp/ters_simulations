for mode_dir in mode_*; do
    for disp in positive_displacement negative_displacement; do
        for field in field_on zero_field; do
            target_dir="$mode_dir/$disp/$field"
            if [ -d "$target_dir" ]; then
                geom="$target_dir/geometry.in"
		ctrl="$target_dir/control.in"
                if [ -f "$geom" ]; then
                    echo "Modifying $geom"
                    cp "$geom" "$geom.bak"  # backup
                    
                    # Insert silver.in content before the last 2 lines
                    head -n -2 "$geom" > "$geom.tmp"
                    cat /scratch/project_2001912/species_defaults/light/47_Ag_default >> "$ctrl"
		    cat silver.in >> "$geom.tmp"
                    tail -n 2 "$geom" >> "$geom.tmp"
                    mv "$geom.tmp" "$geom"
                fi
            fi
        done
    done
done

