CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230313000000_e20230313235959_p20230314021014_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-14T02:10:14.773Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-13T00:00:00.000Z   time_coverage_end         2023-03-13T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        C   records_fill         ]   records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxm�D�  
@          @��@mp���z�?�Q�A��C�T{@mp����?O\)A0(�C��                                    Bxm�Sf  T          @���@l�Ϳ��\?ٙ�A�  C���@l�Ϳ�p�?B�\A$Q�C��=                                    Bxm�b  
�          @���@tz�Tz�?�  A���C���@tz���?��A`��C�`                                     Bxm�p�  
�          @�z�@s33��{?��A�p�C��q@s33��Q�?�  AS
=C�w
                                    Bxm�X  T          @�  @e��\@\)AC�~�@e���R?�=qA_�C���                                    Bxm���  "          @���@]p�����@�B{C���@]p��)��?���Av�HC�G�                                    Bxm���  T          @��H@hQ쿺�H@��A�C��@hQ��!G�?�  A|  C���                                    Bxm��J  "          @���@r�\���R@��A�{C�N@r�\�#�
?��
Ay�C���                                    Bxm���  �          @�
=@�����@Q�A�33C�>�@���Q�?���Au�C�)                                    Bxm�Ȗ  T          @�=q@������@�
A�=qC��@������?��HA]�C�h�                                    Bxm��<  �          @�z�@����\)@#33A�Q�C�XR@���.�R?�ffAj{C�W
                                    Bxm���  �          @���@�����\@(��A��HC�<)@����9��?�ffAiG�C�4{                                    Bxm��  "          @�(�@�����H@#33A�Q�C��)@���333?�  Aa��C�޸                                    Bxm�.  �          @�(�@z�H��\)@.{B Q�C�B�@z�H�A�?��Am�C�&f                                    Bxm��  
�          @�{@z=q��(�@333BG�C��f@z=q�I��?���Al  C���                                    Bxm� z  T          @��R@p  �
=@;�B	�C�Y�@p  �U?�\)As
=C�&f                                    Bxm�/   
�          @�\)@w���(�@8��B��C���@w��Mp�?�33Ay��C�.                                    Bxm�=�  
�          @�  @w
=���@@��BG�C��@w
=�L(�?ǮA�  C�5�                                    Bxm�Ll  T          @�  @vff���H@>�RB
G�C�}q@vff�P  ?�p�A��C��\                                    Bxm�[  �          @�\)@xQ��(�@8��B�C���@xQ��Mp�?�z�AyC�8R                                    Bxm�i�  
�          @�Q�@}p���=q@:�HB�C���@}p��Fff?�  A�33C��q                                    Bxm�x^  �          @�Q�@�p���\)@1G�A��C�^�@�p��6ff?��RA�33C���                                    Bxm��  �          @�z�@A��&ff@N�RB�C��@A��z�H?�{AuG�C��{                                    Bxm���  �          @��@C�
�*=q@L��BQ�C�y�@C�
�}p�?��Aj=qC��R                                    Bxm��P  
�          @�z�@G��'�@H��BffC���@G��x��?��\Ad��C�]q                                    Bxm���  
�          @��R@E�*=q@O\)B
=C���@E�~{?��An�HC��\                                    Bxm���  :          @��R@H���%�@O\)B{C�J=@H���z=q?���Av{C�h�                                    Bxm��B  �          @��R@G��)��@N{B�HC��@G��}p�?�=qAl��C��                                    Bxm���  �          @�{@C�
�,(�@N{B33C�Z�@C�
�~�R?��Aip�C��                                    Bxm��  
          @�
=@J=q�'�@N{B�C�*=@J=q�{�?���Ao�C�g�                                    Bxm��4  �          @�\)@N{�"�\@O\)B�C���@N{�w
=?�z�Ay�C���                                    Bxm�
�  
�          @�
=@S�
�\)@J=qB\)C���@S�
�r�\?�\)As\)C��R                                    Bxm��  
(          @�\)@U��\)@J=qB�HC���@U��q�?�\)As
=C��3                                    Bxm�(&  "          @�ff@W����@G
=B��C��{@W��n{?��Ao33C�                                    Bxm�6�  T          @�  @W
=�   @J�HB�RC��\@W
=�s33?���As\)C��                                     Bxm�Er  �          @���@XQ�� ��@N{B�C��R@XQ��u�?�z�Aw\)C��R                                    Bxm�T  
�          @���@W��(�@P��BG�C��@W��r�\?�p�A�{C��=                                    Bxm�b�  �          @�ff@XQ��\)@O\)B��C�33@XQ��g�?���A�
=C��                                    Bxm�qd  "          @��
@U���
@S33B ��C�(�@U��_\)?�(�A�\)C��R                                    Bxm��
  �          @�z�@W
=�Q�@P  B�HC��\@W
=�aG�?��A���C��{                                    Bxm���  �          @��@W��Q�@L(�B��C�޸@W��_\)?˅A�33C��                                    Bxm��V  T          @��
@[����@I��B�C���@[��^�R?�ffA��HC�J=                                    Bxm���  �          @��
@Y���  @EB�RC�K�@Y���b�\?���A���C��{                                    Bxm���  �          @�(�@Z=q�\)@G�B��C�XR@Z=q�b�\?�(�A��C��\                                    Bxm��H  T          @��
@\(��\)@C33Bp�C�s3@\(��`��?�A���C�4{                                    Bxm���  
�          @�33@aG���R@<(�B��C��\@aG��\(�?�=qAq�C��3                                    Bxm��  
�          @��@`���  @<��B�HC���@`���^{?���Ap(�C��\                                    Bxm��:  
�          @��@c33�{@;�B��C��R@c33�[�?�=qApz�C���                                    Bxm��  �          @�33@a���@=p�B��C�%@a��Z=q?�\)Ay�C��                                    Bxm��  �          @��
@^{��R@B�\B�C��H@^{�_\)?�A��HC�e                                    Bxm�!,  
          @��
@Y�����@HQ�BC���@Y���`��?�G�A�p�C�f                                    Bxm�/�  
@          @�p�@U��R@P  B�C��@U�fff?���A�(�C�e                                    Bxm�>x  �          @�z�@Q��\)@P��B�\C��
@Q��g
=?�{A�C�)                                    Bxm�M  T          @�(�@S33��\@L(�B�RC��q@S33�g�?\A�C�,�                                    Bxm�[�  
Z          @�(�@XQ����@HQ�B\)C��@XQ��c�
?�p�A���C��)                                    Bxm�jj  
�          @�z�@W���\@HQ�B{C��H@W��e?�(�A��HC��
                                    Bxm�y  �          @��H@U��\@E�B{C���@U�dz�?�
=A��RC��{                                    Bxm���  �          @��@XQ��
�H@I��B�C���@XQ��_\)?ǮA�  C�                                    Bxm��\  �          @��@Z=q��@G
=B\)C��3@Z=q�^�R?\A��\C�.                                    Bxm��  T          @��@Vff�=q@AG�B��C�"�@Vff�hQ�?���Ap  C�W
                                    Bxm���  
�          @�33@P  ��@FffB\)C��3@P  �l(�?�\)AyC���                                    Bxm��N  :          @��@P���p�@E�B=qC�z�@P���l��?���At��C��=                                    Bxm���  
�          @��
@N{�!�@Dz�BQ�C���@N{�p��?�ffAl  C�Q�                                    Bxm�ߚ  �          @��H@U��ff@W
=B&=qC��
@U�R�\?�Q�A��C���                                    Bxm��@  	`          @�(�@Z=q��{@UB"C���@Z=q�Tz�?��A���C��                                    Bxm���  "          @�z�@[��ff@Mp�BffC�@ @[��]p�?�33A��C�Z�                                    Bxm��  
�          @�(�@Y�����@H��B
=C��f@Y���`��?��A���C��                                    Bxm�2  �          @��H@Z=q��\@?\)BffC�{@Z=q�`��?�\)Az�\C�R                                    Bxm�(�  "          @��
@U��ff@QG�B��C�ٚ@U��`  ?��HA�C��=                                    Bxm�7~  
�          @���@Y������@fffB,  C��f@Y���[�@Q�A���C�e                                    Bxm�F$  
Z          @���@X�ÿ�{@g�B,p�C��q@X���^{@Q�A��
C�+�                                    Bxm�T�  "          @���@]p�����@`��B%�HC�K�@]p��^�R?��RA���C�c�                                    Bxm�cp  
�          @�G�@X���
�H@[�B!�\C���@X���hQ�?�A��C���                                    Bxm�r  �          @�Q�@W
=�@S�
B��C���@W
=�n{?У�A��
C�                                    Bxm���  �          @�
=@W
=�(�@I��B��C��@W
=�n{?�
=A33C��                                    Bxm��b  
�          @�{@Vff��R@EB=qC��)@Vff�n{?�{At  C��q                                    Bxm��  �          @�ff@Tz��p�@H��B
=C���@Tz��n�R?�A~{C���                                    Bxm���  �          @��R@O\)�,��@B�\B33C�q@O\)�xQ�?��HAX  C��\                                    Bxm��T  "          @���@O\)�'�@A�B�
C�~�@O\)�s�
?��RA^�\C�1�                                    Bxm���  �          @�33@P���'�@:=qB\)C���@P���p  ?��AO\)C�|)                                    Bxm�ؠ  T          @��@Q��$z�@7�B
33C��q@Q��k�?��AP  C��                                     Bxm��F  �          @���@U�$z�@0��B��C�9�@U�g�?��A?33C�XR                                    Bxm���  �          @�=q@^�R�$z�@(��A��HC�@^�R�c�
?p��A*�RC�(�                                    Bxm��  
�          @���@U�#�
@.{B�C�L�@U�e?��\A;�
C�y�                                    Bxm�8  �          @��@8���.�R@G
=B
=C�J=@8���|(�?�G�Ag�
C��                                    Bxm�!�  �          @�33@<(��*=q@L��B��C��=@<(��{�?���A{\)C�g�                                    Bxm�0�  
�          @��@7
=�1�@G�B{C��3@7
=�~�R?�  AeG�C��q                                    Bxm�?*  T          @�=q@0  �C33@>{B=qC�
=@0  ��z�?z�HA2�HC���                                    Bxm�M�  
�          @��H@1G��G
=@9��Bz�C��3@1G����?fffA#33C���                                    Bxm�\v  "          @��@:�H�>�R@8Q�B
��C�8R@:�H����?s33A-p�C��3                                    Bxm�k  �          @���@=p��6ff@<(�Bz�C�f@=p��}p�?���AC�C�e                                    Bxm�y�  
          @��H@7
=�<��@@  BG�C�@7
=��=q?��AAG�C���                                    Bxm��h  
�          @���@333�@��@:�HB�\C�l�@333���\?xQ�A1��C�33                                    Bxm��  "          @�G�@-p��E@:=qB��C���@-p���z�?k�A(��C���                                    Bxm���  �          @���@-p��Fff@:=qB�C���@-p����?k�A(��C���                                    Bxm��Z  
�          @���@'
=�U�@0  B��C��@'
=����?(��@�G�C���                                    Bxm��   �          @���@�R�`��@'
=A�ffC��q@�R��33>�@�  C���                                    Bxm�Ѧ  "          @���@\)�aG�@'�A�{C��@\)���>�@���C��                                    Bxm��L  "          @�@?\)�6ff@G�B�C�/\@?\)��G�?��RA^{C�<)                                    Bxm���  T          @��@E��(Q�@L(�B
=C�˅@E��xQ�?�A
=C�9�                                    Bxm���  
�          @��
@:�H�*�H@O\)B(�C���@:�H�|��?�
=A�=qC�E                                    Bxm�>  �          @���@7��$z�@O\)B �C�
@7��w
=?��RA�G�C�W
                                    Bxm��  T          @���@5��,��@H��B  C�0�@5��z�H?��Aw�C��                                    Bxm�)�  "          @��@&ff�8��@G
=BQ�C���@&ff��=q?��HAa�C�P�                                    Bxm�80  �          @���@)���7�@HQ�B�
C�Y�@)����=q?�  Af�RC��\                                    Bxm�F�  T          @���@)���>�R@C�
B�C��
@)�����?���AP(�C�e                                    Bxm�U|  
�          @��@p��K�@3�
B  C���@p���p�?Q�A{C�Ff                                    Bxm�d"  "          @��
@'
=�:�H@8Q�B\)C��f@'
=�~�R?�G�A@Q�C��                                    Bxm�r�  
Z          @��@��C33@3�
B33C��f@�����?c�
A+33C�#�                                    Bxm��n  �          @�(�@ff�^{@-p�B  C���@ff���?
=@�
=C��                                     Bxm��  "          @�=q@Q��\(�@'�B��C�޸@Q���G�?�@���C�1�                                    Bxm���  T          @��H@��P  @1G�B��C��f@���ff?B�\A��C�<)                                    Bxm��`  "          @�33@{�Q�@333B�HC�f@{���?E�Ap�C���                                    Bxm��  "          @�=q@�
�R�\@7�B�
C�@�
����?Q�Ap�C��{                                    Bxm�ʬ  
�          @���?���j=q@3�
B��C�\?����=q?
=@���C���                                    Bxm��R  �          @�(�?����`  @6ffB��C���?������R?5A�HC�
                                    Bxm���  T          @��\?�(��\(�@:=qB��C�AH?�(���{?L��A  C���                                    Bxm���  	�          @��H?�(��Y��@>�RBG�C�s3?�(���{?aG�A'�C��q                                    Bxm�D  
�          @��?��
�W
=@AG�B�C���?��
��p�?n{A0z�C��
                                    Bxm��  �          @��H?��Tz�@@  B�\C�� ?����
?p��A3�
C�e                                    Bxm�"�  "          @���?���N{@?\)B  C�!H?������?z�HA=��C���                                    Bxm�16  
(          @���?�(��P  @8��B�C�� ?�(���Q�?aG�A(��C�o\                                    Bxm�?�  "          @�=q?��H�X��@3�
B�RC��?��H���H?@  AffC�*=                                    Bxm�N�  "          @��?�(��Y��@1G�B�C�?�(����\?5A
=C�>�                                    Bxm�](  "          @��@33�HQ�@-p�B��C�33@33���?J=qA=qC��=                                    Bxm�k�  �          @��R@�
�A�@0��Bz�C��@�
��  ?aG�A,��C���                                    Bxm�zt  
B          @��H@
=�X��@0��B
p�C��@
=���?5AC��                                    Bxm��  
�          @��H@���O\)@2�\B�
C�w
@����ff?O\)AffC�*=                                    Bxm���  T          @���@p��E@8Q�B�C�:�@p����
?xQ�A6�RC�j=                                    Bxm��f  �          @�(�@%�9��@<��B�\C��H@%�~�R?�\)ATQ�C��f                                    Bxm��  
�          @��@+��3�
@:�HB�RC��{@+��xQ�?��AY�C�L�                                    Bxm�ò  �          @�33@*�H�3�
@9��B
=C�Ǯ@*�H�xQ�?���AW33C�K�                                    Bxm��X  T          @�\)@!��&ff@A�B=qC��@!��p��?��A�{C���                                    Bxm���  "          @�
=@   �*�H@?\)B{C��3@   �r�\?��
A{�
C���                                    Bxm��  
�          @�
=@#33�(��@<��B(�C��q@#33�p��?�G�Ax��C�
                                    Bxm��J  �          @��@���&ff@A�B!��C�Y�@���p��?���A�z�C�L�                                    Bxm��  
(          @�G�@:�H�33@AG�Bp�C�ٚ@:�H�_\)?��RA�p�C��                                    Bxm��  
�          @���@<(����@C�
B  C��R@<(��Z�H?�=qA�{C�Y�                                    Bxm�*<  "          @���@>�R�33@HQ�B#33C���@>�R�Tz�?��HA�C���                                    Bxm�8�  o          @��@<���z�@E�B!��C�o\@<���Tz�?�z�A�(�C��\                                    Bxm�G�  9          @��R@5��R@A�B   C��R@5�[�?�ffA�G�C��
                                    Bxm�V.  
�          @�@,(���@B�\B"  C���@,(��aG�?�G�A�ffC���                                    Bxm�d�  
u          @���@5��
=q@>�RB�C�N@5��Vff?��A��RC�)                                    Bxm�sz  
?          @�ff@-p���@A�B =qC�ff@-p��c33?�p�A��\C���                                    Bxm��   �          @�ff@.{��
@C33B"33C��)@.{�`  ?��
A���C��=                                    Bxm���  T          @�@,���33@B�\B"ffC��f@,���`  ?��
A���C��R                                    Bxm��l  
�          @�ff@*=q�@EB$�\C�S3@*=q�c33?ǮA��HC�g�                                    Bxm��  "          @��
@'
=���@E�B&�C���@'
=�^{?�=qA�Q�C�q�                                    Bxm���  �          @�p�@%��\@I��B)33C�C�@%�a�?У�A�p�C�!H                                    Bxm��^  
�          @�{@#�
�	��@R�\B233C��
@#�
�^�R?�=qA�p�C�'�                                    Bxm��  �          @�
=@!G��@6ffB��C���@!G��[�?���A���C�!H                                    Bxm��  �          @�z�@\)���@-p�B\)C��@\)�Z=q?���A~�\C��                                    Bxm��P  
(          @�Q�@(��   @7
=BffC�#�@(��e�?��A��HC�%                                    Bxm��  T          @���@ ��� ��@333B�C���@ ���c33?�p�A}��C���                                    Bxm��  �          @��@(��(�@?\)B#z�C�w
@(��e�?�
=A�{C�)                                    Bxm�#B  
�          @��@\)��R@>{B �C��@\)�g
=?�33A�p�C�P�                                    Bxm�1�  �          @�G�@%��\)@K�B&�C��@%��mp�?˅A��\C�b�                                    Bxm�@�  �          @��H@)��� ��@K�B$�C�>�@)���n�R?���A�\)C��=                                    Bxm�O4  �          @��H@(���\)@L��B%��C�S3@(���n�R?���A��RC���                                    Bxm�]�  
Z          @�(�@&ff���@Tz�B+��C�N@&ff�p  ?�p�A��C�Y�                                    Bxm�l�  
�          @�z�@#33�   @UB,�C��@#33�s33?�p�A��C��=                                    Bxm�{&  T          @�(�@"�\�{@VffB-�\C���@"�\�q�?�  A�  C��                                    Bxm���  "          @��H@#33��H@U�B-��C�=q@#33�n{?�G�A�{C�5�                                    Bxm��r  �          @���@   �(�@R�\B,��C��R@   �n{?��HA�\)C���                                    Bxm��  
�          @���@(��(�@R�\B.ffC�y�@(��n{?�(�A���C��H                                    Bxm���  "          @�  @ff�{@S33B0�C�Ф@ff�p  ?�(�A�(�C��                                    Bxm��d  "          @�
=@����@P��B.p�C���@��j=q?��HA�(�C���                                    Bxm��
  �          @��@����H@R�\B/��C�Q�@���l��?�p�A���C�s3                                    Bxm��  �          @�ff@=q�(�@Mp�B,\)C�Q�@=q�j�H?�z�A��C��                                     Bxm��V  �          @�ff@�����@N�RB-��C�%@���l(�?�
=A���C�p�                                    Bxm���  T          @�  @{�=q@P��B-��C���@{�j�H?�(�A�  C��3                                    Bxm��  
Z          @�\)@�H�Q�@R�\B0�C��H@�H�j=q?�G�A�p�C��                                     Bxm�H  �          @�
=@�\�p�@S�
B233C�t{@�\�o\)?޸RA�p�C��)                                    Bxm�*�  
�          @�p�@
=��@O\)B0\)C�k�@
=�hQ�?�p�A�(�C��                                    Bxm�9�  �          @�33@	�����@O\)B3ffC��H@	���l(�?�Q�A���C�R                                    Bxm�H:  �          @��H@
�H���@P  B4p�C��@
�H�j=q?�p�A���C�]q                                    Bxm�V�  "          @�33@
�H�=q@P��B4�C��)@
�H�j�H?�p�A��HC�N                                    Bxm�e�  �          @�=q@Q��(�@Mp�B2�RC��@Q��j�H?�A�{C��                                    Bxm�t,  
�          @���@{���@G�B.��C�T{@{�e?�\)A���C���                                    Bxm���  	�          @��H@
�H���@N{B2  C��q@
�H�k�?�
=A��C�AH                                    Bxm��x  
�          @��@
�H���@J�HB0�C�� @
�H�j=q?��A���C�Y�                                    Bxm��  �          @�G�@(��{@G�B-G�C�@(��i��?˅A���C�}q                                    Bxm���  
�          @�Q�@p��\)@C33B*
=C�˅@p��hQ�?\A��
C���                                    Bxm��j  �          @�@(��'
=@N{B-ffC��)@(��s�
?�\)A�\)C���                                    Bxm��  T          @�
=@
=q�'�@Q�B/�C��R@
=q�w
=?�A�\)C��                                     Bxm�ڶ  
�          @�@��(��@O\)B.��C�\)@��vff?У�A�z�C�g�                                    Bxm��\  
�          @��
@�
�*�H@I��B,=qC�Ǯ@�
�u?��A��C�
                                    Bxm��  �          @�(�@�\�*�H@K�B-z�C��{@�\�u?ǮA��
C��q                                    Bxm��  T          @�G�?�
=�,(�@FffB,p�C���?�
=�tz�?��RA�33C�g�                                    Bxm�N  T          @���?�33�,��@E�B,=qC��?�33�u�?�(�A�  C�+�                                    Bxm�#�  �          @��R?�=q�)��@E�B.�RC�S3?�=q�q�?��RA�ffC��\                                    Bxm�2�  
�          @�?��'
=@E�B0�RC�Z�?��o\)?\A��\C��                                    Bxm�A@  
�          @��?�(��)��@Dz�B0p�C���?�(��qG�?��RA���C�G�                                    Bxm�O�  �          @��?�\�%@@��B/33C�*=?�\�l��?�(�A��C��                                    Bxm�^�  
�          @��?���#�
@>{B.=qC��H?���h��?���A�
=C�R                                    Bxm�m2  �          @�Q�?��
�"�\@:�HB,�HC�|)?��
�fff?�z�A���C�"�                                    Bxm�{�  �          @��?�p��%�@8��B,
=C��f?�p��g�?���A�\)C��)                                    Bxm��~  "          @���?�  �#33@=p�B/�C�33?�  �hQ�?���A�z�C��R                                    Bxm��$  "          @�Q�?�p��!G�@>{B0�C�/\?�p��g
=?�(�A�C�Ǯ                                    Bxm���  T          @�  ?����"�\@7�B*z�C�˅?����e�?���A�{C�w
                                    Bxm��p  
�          @��?��#�
@8Q�B(
=C�k�?��fff?���A��
C��                                    Bxm��  "          @�Q�?�\��R@>�RB1G�C�Ǯ?�\�dz�?�G�A��C�4{                                    Bxm�Ӽ  
�          @�
=?��H�\)@<��B1(�C�E?��H�c�
?�p�A�  C��
                                    Bxm��b  
�          @�\)?���{@:�HB/33C���?���a�?�(�A���C�o\                                    Bxm��  "          @���?��
�!�@=p�B.�C��
?��
�fff?�(�A�G�C�*=                                    Bxm���  
�          @�ff?�\)�#33@:�HB0{C�>�?�\)�fff?�
=A��C�!H                                    Bxm�T  	�          @�Q�?�33�%@<(�B/(�C�L�?�33�i��?�Q�A�=qC�4{                                    Bxm��  �          @���?�33�(��@<��B-C���?�33�l��?�A��HC��                                    Bxm�+�  
�          @�33?�33�+�@@  B.ffC��)?�33�p  ?���A�  C���                                    Bxm�:F  "          @��
?У��,��@A�B/ffC���?У��q�?�(�A�\)C��=                                    Bxm�H�  T          @�(�?�33�,(�@A�B/ffC���?�33�qG�?�p�A�(�C��=                                    Bxm�W�  �          @��H?У��+�@@��B/=qC��?У��p  ?�(�A��C�                                    Bxm�f8  �          @��H?���,(�@B�\B1C���?���qG�?��RA��HC��                                    Bxm�t�  "          @�33?����,(�@AG�B/�C�XR?����qG�?�p�A��HC�}q                                    Bxm���  
Z          @��
?���*=q@C33B1Q�C��)?���p  ?\A�G�C��{                                    Bxm��*  "          @�33?�z��(��@B�\B1{C��?�z��n{?\A��C��                                    Bxm���  
�          @���?��&ff@?\)B0
=C�]q?��j�H?�  A��HC�AH                                    Bxm��v  T          @�33?�
=�(��@AG�B/�C�B�?�
=�n{?�G�A��\C�.                                    Bxm��  T          @���?�{�:�H@C�
B*p�C�n?�{�~�R?�
=A�
=C��q                                    Bxm���  "          @�?Ǯ�8��@>{B(��C�'�?Ǯ�z=q?�{A��C��
                                    Bxm��h  "          @��
?��H�8Q�@<(�B)(�C�e?��H�x��?�=qA�z�C�@                                     Bxm��  �          @��H?��R�5@:=qB(�
C��)?��R�vff?�=qA�G�C��q                                    Bxm���  T          @��
?�Q��;�@8��B&=qC�?�Q��z�H?��
A�z�C�3                                    Bxm�Z  T          @��?�\)�>�R@7�B$�
C�XR?�\)�|��?��RA�Q�C���                                    Bxm�   �          @��
?�33�<��@G�B*C���?�33��G�?�p�A��
C��                                    Bxm�$�  T          @�ff?�{�6ff@O\)B.ffC���?�{�\)?��A���C��f                                    Bxm�3L  
Z          @�\)?�Q��5�@P  B-��C�8R?�Q��~�R?�z�A�=qC��q                                    Bxm�A�  
C          @�p�?����1�@P��B1p�C��3?����|(�?�Q�A�  C��{                                    Bxm�P�  
q          @�G�?��
�.{@K�B1�\C��{?��
�u?��A�C�e                                    Bxm�_>  �          @��
?��
�1G�@O\)B2�C�h�?��
�z=q?�
=A���C�AH                                    Bxm�m�            @�ff?�G��>{@L(�B+  C�` ?�G���=q?�ffA�(�C��R                                    Bxm�|�  
�          @�p�?޸R�8Q�@N�RB/33C��=?޸R��Q�?�\)A�
=C���                                    Bxm��0  T          @�{?У��;�@Q�B1G�C���?У����\?�33A���C��                                    Bxm���  "          @���?���@��@L(�B-33C��f?�����?��A�ffC�C�                                    Bxm��|  �          @��H?�p��0  @Mp�B2�RC�\?�p��xQ�?�
=A�=qC��q                                    Bxm��"  T          @�G�?���'�@O\)B6(�C�.?���q�?�G�A�ffC���                                    Bxm���  T          @���?ٙ��,��@Mp�B4��C�  ?ٙ��u�?��HA�G�C��R                                    Bxm��n  �          @���?�33�0  @K�B3  C�|)?�33�w�?�z�A�  C���                                    Bxm��  �          @�G�?�(��.�R@K�B2ffC�3?�(��vff?�A���C�f                                    Bxm��  �          @�G�@��$z�@H��B/=qC�q@��l(�?ٙ�A�z�C�ff                                    Bxm� `  �          @���@
=q� ��@Dz�B+
=C�\)@
=q�fff?�A�
=C��H                                    Bxm�  "          @�G�@��(Q�@C�
B)z�C�/\@��l��?�{A�=qC��\                                    Bxm��  
�          @��
@z��+�@H��B+z�C��3@z��q�?�A��C�XR                                    Bxm�,R  T          @�(�@��.�R@FffB(ffC��@��s33?�{A��HC�S3                                    Bxm�:�  �          @��H?����*�H@J�HB.�C��?����q�?ٙ�A�C��H                                    Bxm�I�  �          @�33?��.�R@N�RB2\)C��?��w
=?�p�A��RC���                                    Bxm�XD  �          @�33?�=q�,��@P  B3��C��?�=q�u?�G�A�(�C��
                                    Bxm�f�  T          @��H?���*�H@N{B2\)C���?���s33?�G�A�(�C�B�                                    Bxm�u�  
�          @��H?����%@P  B4�C���?����n�R?���A�ffC�Ф                                    Bxm��6  �          @��@�\�'�@HQ�B-ffC��
@�\�mp�?ٙ�A�33C�ff                                    Bxm���  "          @��H@	���(Q�@EB)�C���@	���l��?�z�A��C��                                    Bxm���  "          @�33@���,(�@C�
B&�C�Ff@���o\)?�{A���C��                                    Bxm��(  
�          @���@��*=q@C33B(�C�  @��mp�?�\)A���C��                                     Bxm���  �          @��@(��"�\@E�B*G�C�\)@(��g�?ٙ�A��C��
                                    Bxm��t  "          @���@Q��$z�@C�
B)�C�Ф@Q��hQ�?�A���C�1�                                    Bxm��  �          @�=q@z��+�@Dz�B(�C���@z��o\)?У�A�
=C��                                     Bxm���  "          @��?�p��@��@A�B$Q�C���?�p���Q�?��HA���C��=                                    Bxm��f  "          @��\?�\�8Q�@EB)C���?�\�z�H?���A���C�,�                                    Bxm�  
�          @��H?����1�@C33B'��C�~�?����s�
?˅A��\C��                                    Bxm��  �          @��?�(��2�\@@  B%=qC��
?�(��s33?��A��C���                                    Bxm�%X  �          @�G�@z��-p�@?\)B$�C���@z��n{?ǮA�
=C��f                                    Bxm�3�  
�          @�=q?��R�4z�@?\)B#�C��H?��R�tz�?\A���C�Ǯ                                    Bxm�B�  �          @��\@ ���333@@��B$z�C�ٚ@ ���s�
?�ffA�(�C���                                    Bxm�QJ  
�          @�{@ ���:�H@E�B$  C�H�@ ���|��?���A��C��H                                    Bxm�_�  
�          @��H@��>�R@L(�B%Q�C�k�@�����?��A���C���                                    Bxm�n�  T          @�  @�5@L(�B(ffC�(�@�z�H?ٙ�A���C��                                    Bxm�}<  T          @��?���<��@@��B"�RC�K�?���|(�?��RA��C��=                                    Bxm���  �          @�z�?��H�;�@AG�B"�C��H?��H�z�H?\A���C�@                                     Bxm���  T          @�G�?���8��@<��B"�C�y�?���w
=?�(�A��C��{                                    Bxm��.  �          @�G�?�  �333@/\)B��C��
?�  �l(�?���A�C���                                    Bxm���  T          @�=q?����5�@.{B  C�^�?����l��?�ffA��C��                                    Bxm��z  
�          @�z�?�  �:=q@1G�B�C�~�?�  �s33?��A���C�\)                                    Bxm��   "          @���?�33�7�@0  B��C�Ǯ?�33�p  ?��A��RC�p�                                    Bxm���  
�          @�p�?�ff�@  @,��B�RC��H?�ff�vff?��HA33C���                                    Bxm��l  �          @�
=?����I��@0  B��C�xR?�����  ?���Ax��C���                                    Bxm�  �          @��?�=q�AG�@2�\B{C��H?�=q�z=q?��A�  C���                                    Bxm��  �          @�(�?�
=�:=q@333B�C��)?�
=�s33?���A�Q�C���                                    Bxm�^  
D          @�{?�=q�8��@5BC�(�?�=q�s33?��A�Q�C��
                                    Bxm�-  8          @��?���7
=@0��B
=C�?���o\)?��A��HC���                                    Bxm�;�  
�          @��\?�33�<��@,��B=qC���?�33�s33?��RA�C���                                    Bxm�JP  T          @��?��H�6ff@4z�B!Q�C�u�?��H�p��?�33A�=qC�8R                                    Bxm�X�  "          @��?��8Q�@3�
B ��C��?��qG�?��A�z�C��                                    Bxm�g�  �          @��?�
=�.�R@1�B${C��=?�
=�hQ�?�
=A��C�g�                                    Bxm�vB  "          @���?�{�(��@5�B$��C��
?�{�dz�?�G�A�  C���                                    Bxm���  �          @��@ ���&ff@<(�B'
=C���@ ���dz�?�\)A���C��
                                    Bxm���  
�          @�p�@G��#�
@>�RB)z�C�'�@G��c33?�
=A��\C���                                    Bxm��4  "          @��@  �(�@AG�B)�C�U�@  �\��?�\A�=qC���                                    Bxm���  T          @���@�R���@:=qB p�C��
@�R�W�?�
=A�33C�(�                                    Bxm���  T          @�  @(���  @7
=B{C��f@(���N{?ٙ�A�=qC���                                    Bxm��&  "          @��@'����@7
=B�C���@'��N{?��HA���C���                                    Bxm���  
�          @�  @(Q��p�@:=qB!  C�  @(Q��L(�?�\A�33C��q                                    Bxm��r  T          @�=q@*=q�{@>�RB"��C��@*=q�N�R?�=qA���C���                                    Bxm��  �          @�=q@%�
=@G�B,(�C�h�@%�L(�@ ��A�ffC���                                    Bxm��  T          @��@#33�G�@L��B2Q�C��\@#33�HQ�@�A�C���                                    Bxm�d  
�          @�G�@   �   @N�RB5{C���@   �G�@
=qA��C�T{                                    Bxm�&
  	�          @��@\)��
@N{B3C�(�@\)�K�@Q�A�ffC�H                                    Bxm�4�  �          @�z�@'���z�@S�
B6ffC���@'��Dz�@G�A��HC�<)                                    Bxm�CV  T          @�p�@'�����@X��B:33C�l�@'��C33@�A�  C�`                                     Bxm�Q�  �          @�p�@5�ٙ�@R�\B3z�C�� @5�8Q�@A��HC�N                                    Bxm�`�  T          @���@0�׿�\@R�\B4�C��3@0���;�@z�A�C��f                                    Bxm�oH  �          @��H@+���p�@S�
B8
=C��\@+��9��@ffA��C�\)                                    Bxm�}�  
�          @�(�@*=q��p�@W
=B:�C�u�@*=q�;�@��A�  C�&f                                    Bxm���  �          @���@,�Ϳ޸R@W
=B9�C��q@,���;�@��A�Q�C�Z�                                    Bxm��:  
�          @���@0�׿���@XQ�B:��C��
@0���333@{B ��C�C�                                    Bxm���  
�          @��
@$z��@XQ�B<ffC�s3@$z��@  @��A�  C�S3                                    Bxm���  �          @��@�R��@Z�HB?z�C���@�R�B�\@=qA��HC��)                                    Bxm��,  "          @�p�@%��=q@Z�HB<��C�Z�@%�A�@�A��HC�@                                     Bxm���  T          @��@"�\��@\��B?�C��@"�\�B�\@��A�(�C��                                    Bxm��x  �          @�z�@'
=��  @Z�HB=�C�{@'
=�<��@p�B   C��                                     Bxm��  T          @�@,�Ϳ�@\(�B=
=C�'�@,���8Q�@ ��BC��
                                    Bxm��  T          @�
=@*�H��
=@aG�B@��C���@*�H�:�H@%�B�RC�7
                                    Bxm�j  �          @��@1G��˅@`  B>=qC�@1G��5�@&ffB{C�4{                                    Bxm�  �          @�Q�@/\)�\@e�BCQ�C�� @/\)�2�\@-p�B
��C�9�                                    Bxm�-�  �          @���@/\)���
@fffBCC�ff@/\)�3�
@.{B{C�!H                                    Bxm�<\  T          @�  @=p��˅@k�B>G�C��)@=p��9��@1G�B33C��3                                    Bxm�K  
�          @���@B�\��\)@l��B<(�C���@B�\�:�H@1�B�C�                                    Bxm�Y�  "          @�\)@AG����
@i��B<��C���@AG��4z�@1�B�C�xR                                    Bxm�hN  
�          @�@W�����@�  B>  C�Q�@W��8��@HQ�B�HC��3                                    Bxm�v�  T          @�Q�@\(����H@���B=
=C���@\(��9��@J�HB�C��\                                    Bxm���  �          @��R@]p����@�  B<�\C�ff@]p��1�@K�BffC��                                     Bxm��@  �          @�z�@X�ÿ���@}p�B=z�C��@X���1G�@H��B��C�Z�                                    Bxm���  
�          @��@\(����\@y��B;(�C��H@\(��*�H@G�Bz�C��                                    Bxm���  
�          @��@Mp����H@p  B;33C��H@Mp��2�\@:=qB	��C���                                    Bxm��2  �          @��H@C�
��(�@k�B9�C�O\@C�
�@  @0  B�C���                                    Bxm���  �          @�33@Dz��\@j�HB7�RC��@Dz��B�\@-p�B z�C���                                    Bxm��~  �          @�33@Fff��=q@g
=B4�C�@Fff�C�
@)��A�=qC���                                    Bxm��$  T          @��@@  �   @g�B4p�C�/\@@  �N{@%A�C�}q                                    Bxm���  �          @���@@  ��(�@l(�B7{C�XR@@  �N{@*=qA�p�C�y�                                    Bxm�	p  �          @�(�@@�׿��R@h��B5{C�G�@@���N{@'�A�C���                                    Bxm�  T          @�z�@C�
��p�@g�B3(�C��3@C�
�L��@&ffA�C��H                                    Bxm�&�  "          @�z�@Fff��@g�B3{C�#�@Fff�H��@(Q�A�{C�S3                                    Bxm�5b  "          @��
@G
=����@dz�B0��C��@G
=�I��@$z�A�p�C�H�                                    Bxm�D  �          @��H@Dz��33@eB3�C�#�@Dz��G
=@'
=A��\C�W
                                    Bxm�R�  �          @��
@E���G�@l��B8�RC�  @E��AG�@1G�B�RC���                                    Bxm�aT  
�          @��
@Fff�У�@n{B:�
C��@Fff�:=q@5B�C�l�                                    Bxm�o�  T          @�p�@G
=��(�@p��B:��C���@G
=�@  @6ffB�C��q                                    Bxm�~�  
�          @�p�@G
=��p�@o\)B9�C�o\@G
=�@��@5�B��C���                                    Bxm��F  T          @�p�@J=q���H@mp�B7�C���@J=q�>{@333Bz�C�h�                                    Bxm���  
�          @�{@I�����
@n{B7
=C�N@I���A�@2�\B�C��                                    Bxm���  
�          @�ff@I������@l��B5ffC��@I���Fff@0  A��C��H                                    Bxm��8  
�          @�@HQ���@j�HB4G�C�|)@HQ��G
=@-p�A�{C��)                                    Bxm���  �          @��@G
=��=q@l(�B6\)C��@G
=�Dz�@0  B �C��R                                    Bxm�ք  
�          @�{@HQ��@n�RB7z�C�H@HQ��C�
@333BQ�C�ٚ                                    Bxm��*  T          @���@N{��@p��B5C��@N{�Fff@4z�B{C��                                    Bxm���  �          @���@N{���@n{B3��C��\@N{�HQ�@1G�A��C��                                    Bxm�v  �          @���@Mp����H@l(�B1�C�E@Mp��K�@-p�A��C��q                                    Bxm�  "          @���@Mp����H@l(�B1��C�J=@Mp��K�@-p�A�  C��                                    Bxm��  �          @���@N�R� ��@j=qB/�RC��@N�R�Mp�@*�HA�\)C��3                                    Bxm�.h  T          @���@N{�G�@j�HB/��C��3@N{�N{@*�HA�C�~�                                    Bxm�=  �          @���@O\)��(�@j=qB/�
C�XR@O\)�J�H@,(�A�G�C�Ф                                    Bxm�K�  T          @�
=@L���@b�\B+{C�g�@L���O\)@!�A�RC�U�                                    Bxm�ZZ  "          @�ff@N{�
=@_\)B(�HC�ff@N{�N�R@\)A���C�q�                                    Bxm�i   �          @�{@O\)��@Z�HB$�HC��@O\)�QG�@��A�{C�]q                                    Bxm�w�  T          @��@G��
=@a�B,��C��{@G��O\)@!�A�ffC��                                    Bxm��L  T          @�
=@K��p�@`  B(p�C���@K��Tz�@p�A�C��                                     Bxm���  �          @�{@N�R�\)@X��B#�C���@N�R�S�
@ffA��
C�%                                    Bxm���  �          @�@P���{@VffB!Q�C�ٚ@P���Q�@z�A�p�C�h�                                    Bxm��>  
�          @��@R�\�
=q@^{B%��C�^�@R�\�P��@p�A�Q�C���                                    Bxm���  	�          @���@O\)�Q�@e�B*��C�T{@O\)�QG�@$z�A�p�C�Z�                                    Bxm�ϊ  "          @��R@N{��R@\(�B$�C�� @N{�Tz�@=qA�{C�{                                    Bxm��0  T          @�{@Mp���R@Y��B${C��)@Mp��S33@Q�Aڣ�C��                                    Bxm���  
�          @��@J�H���@Z=qB%�HC���@J�H�Q�@��A�=qC��                                    Bxm��|  �          @�33@G
=�
�H@Z�HB(G�C��{@G
=�O\)@�HA��HC���                                    Bxm�
"  �          @��@HQ��(�@Y��B&��C��@HQ��P  @��A��
C���                                    Bxm��  
�          @��
@H����@Z=qB'�C��)@H���P  @�HA�G�C�H                                    Bxm�'n  "          @�(�@E����@Z�HB'(�C�ٚ@E��U�@��Aޣ�C�ff                                    Bxm�6  "          @�33@B�\�{@\(�B)p�C��f@B�\�S33@�A�\)C�XR                                    Bxm�D�  �          @��\@J=q��@Y��B'��C�T{@J=q�H��@(�A�{C���                                    Bxm�S`  
�          @���@K���@VffB&Q�C���@K��E�@�HA�33C���                                    Bxm�b  
�          @��H@N�R��\@W
=B%G�C��H@N�R�E@�A�  C�%                                    Bxm�p�  T          @���@Mp���@^�RB*G�C��
@Mp��G�@#33A�p�C��H                                    Bxm�R  
�          @�(�@I��� ��@aG�B-33C��R@I���G�@%A�RC��                                    Bxm���  T          @�z�@G
=��\@c33B.�C�c�@G
=�I��@'
=A�ffC�Q�                                    Bxm���  �          @���@E���@fffB133C�Q�@E��J=q@*=qA��C�%                                    Bxm��D  �          @�(�@AG���\@g
=B2�HC��R@AG��J�H@*�HA��C�˅                                    Bxm���  
�          @�@AG��Q�@hQ�B2
=C�\)@AG��P��@*=qA�p�C�^�                                    Bxm�Ȑ  �          @�\)@C�
�@l(�B3p�C�ٚ@C�
�O\)@/\)A�(�C���                                    Bxm��6  
�          @�  @C33�
=@n{B4\)C��H@C33�QG�@0��A�G�C�|)                                    Bxm���  �          @���@E��
�H@l(�B1�C�l�@E��S�
@-p�A�p�C�xR                                    Bxm��  �          @���@C�
��R@j=qB0(�C���@C�
�W
=@*�HA�\)C�,�                                    Bxm�(  
�          @�  @?\)�33@j=qB0��C�>�@?\)�Z�H@(��A�C��R                                    Bxm��  �          @�
=@=p���R@j�HB2��C�~�@=p��W
=@+�A�\)C���                                    Bxm� t  �          @��@:�H�33@l(�B3=qC���@:�H�[�@+�A��C�7
                                    Bxm�/  
�          @�Q�@;����@qG�B7\)C���@;��W
=@2�\B 33C��                                    Bxm�=�  �          @�p�@A녿�@n{B8Q�C��@A��E�@5BQ�C�<)                                    Bxm�Lf  �          @���@A녿���@n{B9\)C�G�@A��AG�@7�B\)C���                                    Bxm�[  
Z          @�@AG��\)@c�
B-�C��
@AG��Tz�@%�A�\)C�&f                                    Bxm�i�  �          @�{@A���@^{B((�C�H@A��Z=q@p�A�ffC��{                                    Bxm�xX  �          @��@C33�33@^{B(�HC��H@C33�U@�RA��
C�5�                                    Bxm���  
�          @��@A��Q�@b�\B*ffC��\@A��\(�@!�A���C���                                    Bxm���  �          @�  @@  �(�@c33B*�C�t{@@  �_\)@ ��A���C�T{                                    Bxm��J  
�          @���@C�
��H@c33B)G�C��@C�
�^�R@!G�A�RC��f                                    Bxm���  
�          @�  @Fff�Q�@`  B'\)C�33@Fff�Z�H@\)A�RC�\                                    Bxm���  T          @�
=@H���\)@a�B)�
C�=q@H���R�\@$z�A�C�˅                                    Bxm��<  �          @�@E��33@^�RB(�\C��=@E��U�@   A�33C�e                                    Bxm���  "          @�\)@:�H��@P  B#33C�t{@:�H�Tz�@��A��HC��f                                    Bxm��  T          @�p�@5�(�@L(�B"  C��@5�W
=@��A�  C�3                                    Bxm��.  �          @�{@8Q��{@J=qB  C��@8Q��X��@	��A�z�C�33                                    Bxm�
�  �          @��R@:�H�!�@G
=Bz�C��H@:�H�[�@A�\)C�1�                                    Bxm�z  T          @�\)@9���%�@G
=B��C�#�@9���^{@�A�\)C��                                    Bxm�(   
�          @�(�@333�   @FffBp�C�  @333�X��@ffA�=qC�Ǯ                                    Bxm�6�  "          @�  @1G���@Dz�B!�C��R@1G��N{@�A�z�C�aH                                    Bxm�El  �          @�z�@,���p�@C�
B%{C�XR@,���Fff@
=qAݙ�C���                                    Bxm�T  T          @��@*�H�
=q@J=qB*ffC�y�@*�H�E�@G�A�z�C�k�                                    Bxm�b�  �          @�{@/\)��@J=qB)Q�C�/\@/\)�B�\@�A��C��                                    Bxm�q^  �          @��R@.�R�	��@J�HB)\)C��f@.�R�E�@�\A�ffC��3                                    Bxm��  T          @�p�@333�   @`��B6
=C�33@333�B�\@)��B�C�L�                                    Bxm���  
�          @�p�@333� ��@aG�B6��C�+�@333�C33@*�HBG�C�B�                                    Bxm��P  �          @�z�@.�R� ��@a�B8�C���@.�R�C33@+�BC��                                    Bxm���  "          @��R@4z��z�@aG�B4��C��@4z��Fff@)��B�C�&f                                    Bxm���  T          @���@5�   @]p�B3�\C�ff@5�AG�@'�B{C���                                    Bxm��B  �          @�{@5��(�@Y��B.  C�!H@5��K�@   A�p�C�ٚ                                    Bxm���  "          @�Q�@7��G�@Z�HB,G�C���@7��P��@   A��C��=                                    Bxm��  
Z          @���@6ff�ff@Z=qB*�HC�5�@6ff�U�@p�A�RC�AH                                    Bxm��4  
�          @���@7��#33@Q�B!�HC�0�@7��^{@G�A�=qC���                                    Bxm��  "          @�G�@2�\�+�@N{B
=C�R@2�\�e�@�A�p�C���                                    Bxm��  T          @���@0  �,(�@QG�B!Q�C��\@0  �fff@�RA��C��\                                    Bxm�!&  �          @��
@5��0��@P  B��C�ٚ@5��j=q@(�A���C��R                                    Bxm�/�  
�          @��
@5��1�@N�RB�C��=@5��j�H@
�HAȏ\C��R                                    Bxm�>r  �          @�z�@7
=�+�@Tz�B �C�h�@7
=�g
=@�A�
=C�5�                                    Bxm�M  T          @���@8Q��-p�@S33B=qC�ff@8Q��g�@��A��C�E                                    Bxm�[�  
�          @�@:�H�-p�@S�
B�
C���@:�H�hQ�@G�A��
C�ff                                    Bxm�jd  �          @�ff@:=q�.{@UB�HC�o\@:=q�i��@33AѮC�H�                                    Bxm�y
  
�          @�ff@;��*=q@XQ�B"
=C��@;��e@
=Aי�C��{                                    Bxm���  �          @�{@:�H�'
=@Z=qB#��C�"�@:�H�c�
@��A�Q�C��R                                    Bxm��V  �          @�ff@<���$z�@[�B$�C�z�@<���aG�@�A߅C��R                                    Bxm���  "          @�@>{�   @[�B&  C���@>{�\��@p�A�\)C�W
                                    Bxm���  "          @��@8���#33@[�B&��C�K�@8���`  @��A�33C��H                                    Bxm��H  "          @�p�@8Q��"�\@^{B(ffC�T{@8Q��`  @\)A�Q�C��)                                    Bxm���  
�          @��@>{��H@]p�B(�\C�e@>{�X��@!�A�Q�C���                                    Bxm�ߔ  
�          @�@AG����@]p�B(
=C��\@AG��W
=@"�\A��HC��R                                    Bxm��:  
�          @��R@<���p�@`��B)�\C��@<���\(�@#�
A뙚C�O\                                    Bxm���  "          @�ff@>{���@a�B+G�C���@>{�XQ�@'
=A��C��                                     Bxm��  �          @�  @@  �(�@c�
B*z�C�n@@  �[�@'
=A��HC��{                                    Bxm�,  �          @�@A����@dz�B.\)C�  @A��Mp�@,��A��
C��{                                    Bxm�(�  T          @�ff@?\)�p�@hQ�B1p�C��=@?\)�N�R@1G�B ��C�ff                                    Bxm�7x  T          @�
=@<(���R@a�B*=qC���@<(��\��@%�A�p�C�1�                                    Bxm�F  �          @��R@C�
��@g
=B/ffC�G�@C�
�L(�@0��A�p�C��                                    Bxm�T�  
�          @�
=@E��G�@c33B+�\C�˅@E��P��@*�HA���C��\                                    Bxm�cj  
Z          @���@L(��ff@`��B&  C��@L(��Tz�@'
=A�{C��f                                    Bxm�r  �          @���@G��3�
@N{B�C��q@G��j�H@��A�{C�1�                                    Bxm���  
�          @���@Fff�(�@aG�B'  C�� @Fff�Z=q@&ffA�\)C�#�                                    Bxm��\  �          @�Q�@E��z�@dz�B+\)C���@E��S�
@,(�A�  C�|)                                    Bxm��  "          @��@K��33@h��B+G�C�@K��S�
@0  A��
C���                                    Bxm���  T          @��@J�H�#�
@j�HB'�HC��=@J�H�c�
@-p�A�\C��R                                    Bxm��N  �          @�\)@Q��  @p  B-Q�C�@Q��R�\@8��A���C�q�                                    Bxm���  �          @���@Mp��
=@s33B2�HC�W
@Mp��J�H@>�RB�RC���                                    Bxm�ؚ  
�          @��R@W
=��@g�B%�C�� @W
=�Tz�@.�RA�Q�C��=                                    Bxm��@  �          @�{@S33�ff@hQ�B'z�C�L�@S33�U@0  A�z�C�U�                                    Bxm���  �          @�
=@Q����@h��B&z�C���@Q��\(�@.{A�z�C���                                    Bxm��  "          @��@Q��#33@eB#G�C��@Q��`��@)��A�RC��f                                    Bxm�2  "          @��@L���'
=@_\)B �HC�p�@L���b�\@"�\A�{C��                                    Bxm�!�  "          @�{@J�H�333@Z�HB�RC�H�@J�H�l��@=qAҸRC�L�                                    Bxm�0~  �          @�{@L���-p�@\(�B�\C��
@L���hQ�@{A�(�C��{                                    Bxm�?$  �          @��\@G��=p�@I��B�RC�H�@G��qG�@
=A��\C���                                    Bxm�M�            @�Q�@B�\�:�H@I��B�C�3@B�\�n�R@Q�A��HC��{                                    Bxm�\p  
�          @�  @A��<(�@FffB(�C���@A��o\)@�A���C��=                                    Bxm�k  �          @��@G
=�>�R@Q�B{C�q@G
=�tz�@\)A��C��\                                    Bxm�y�  �          @���@I���J�H@O\)B�C�k�@I����  @	��A���C�%                                    Bxm��b  
�          @�\)@C�
�>�R@Z�HB�RC��q@C�
�w�@Q�A�C�,�                                    Bxm��  �          @�{@AG��9��@^{BffC��@AG��s33@(�A�C�>�                                    Bxm���  "          @��@N�R�<(�@U�BG�C��@N�R�s33@33A�ffC�(�                                    Bxm��T  
�          @�
=@G��6ff@]p�B=qC��@G��p  @p�A�{C��                                    Bxm���  T          @�\)@W����@z�HB-(�C�  @W��`  @AG�A�
=C��{                                    Bxm�Ѡ  �          @���@Y���"�\@z=qB*�\C��
@Y���e@>�RA���C���                                    Bxm��F  
�          @��@\���#33@z=qB)ffC��=@\���e@?\)A��C���                                    Bxm���  �          @�G�@^�R�#�
@vffB&�
C�Ф@^�R�e�@:�HA��HC�{                                    Bxm���  �          @�\)@`  � ��@q�B%  C�+�@`  �`��@7�A��C�t{                                    Bxm�8  	�          @��@aG��(Q�@l(�B 33C��q@aG��fff@0��A�C�.                                    Bxm��  �          @�Q�@b�\�.{@i��B�C�@ @b�\�j�H@,(�A���C���                                    Bxm�)�  
�          @��\@dz��2�\@j�HB�C��@dz��o\)@,(�A��
C��R                                    Bxm�8*  "          @��@e�1�@l��B��C�(�@e�o\)@.�RA�(�C��                                    Bxm�F�  �          @��H@dz��3�
@j=qB�C��@dz��p��@+�A�
=C��                                    Bxm�Uv  T          @�ff@^{�.�R@eBffC��f@^{�j=q@(��A���C��                                     Bxm�d  "          @��@_\)�0  @hQ�B{C���@_\)�k�@+�A�ffC��R                                    Bxm�r�  �          @��@Z=q�*=q@p��B${C��)@Z=q�h��@5�A�ffC���                                    Bxm��h  �          @�z�@\(��#�
@�G�B-(�C���@\(��g�@HQ�B �C��                                    Bxm��  
�          @���@g
=�.�R@~�RB%Q�C�n@g
=�qG�@AG�A��\C��                                    Bxm���  �          @\@n�R�Fff@i��B�
C�*=@n�R����@&ffA��HC�s3                                    Bxm��Z  
�          @�33@n{�J=q@j=qBQ�C��3@n{���\@%A�
=C�.                                    Bxm��   
�          @��
@q��J�H@g�B�C�H@q����\@#�
AŅC�h�                                    Bxm�ʦ  �          @�33@p  �5�@xQ�B�C�y�@p  �u�@9��A�p�C�7
                                    Bxm��L  
�          @�z�@q��6ff@y��B�HC���@q��vff@:�HA�G�C�B�                                    Bxm���  "          @Å@n�R�333@|(�B C��\@n�R�s�
@>{A�33C�33                                    Bxm���  �          @��
@mp��2�\@~�RB"��C���@mp��s�
@AG�A���C�                                      Bxm�>  T          @���@mp��0��@��B%
=C���@mp��s33@FffA�{C�!H                                    Bxm��  �          @���@p���1G�@�  B"p�C��R@p���r�\@C33A��
C�c�                                    Bxm�"�  �          @��@p���-p�@��B$C��@p���p  @G�A�G�C���                                    Bxm�10  �          @Å@j=q�/\)@��B&�C���@j=q�q�@G�A�33C��                                    Bxm�?�  �          @��H@l���,��@�Q�B$C��@l���n�R@E�A�
=C�ff                                    Bxm�N|  T          @��
@p  �*�H@���B$�C�H�@p  �l��@G
=A��RC���                                    Bxm�]"  T          @���@vff�%@�G�B$�C��@vff�hQ�@H��A��C�XR                                    Bxm�k�  "          @�z�@xQ��#33@���B#�C�XR@xQ��e�@H��A�  C��f                                    Bxm�zn  
�          @�p�@w
=�%�@�=qB$�HC�)@w
=�g�@K�A��C�g�                                    Bxm��  
�          @�{@u��R@�p�B)p�C��3@u�c33@S33B{C��q                                    Bxm���  �          @���@tz��&ff@�=qB%�\C���@tz��h��@K�A�z�C�,�                                    Bxm��`  T          @�p�@z=q���@�z�B(�C�K�@z=q�\��@S�
B�\C�Ff                                    Bxm��  �          @�ff@|(��,��@|(�B{C��@|(��l��@A�A�(�C�g�                                    Bxm�ì  
�          @�\)@|(��7
=@xQ�B=qC�f@|(��u�@;�A�ffC��=                                    Bxm��R  "          @�\)@{��:�H@vffBC��\@{��xQ�@8Q�A܏\C��                                    Bxm���  �          @Ǯ@|(��@  @s33B(�C�Y�@|(��|(�@4z�AָRC�}q                                    Bxm��  
�          @�\)@|(��>�R@r�\B=qC�t{@|(��z=q@4z�A�\)C���                                    Bxm��D  �          @�  @|���<��@u�B\)C��q@|���y��@7
=A�(�C��\                                    Bxm��  �          @Ǯ@{��<(�@w
=B��C�� @{��x��@8��AܸRC���                                    Bxm��  
�          @�
=@|(��8��@vffB��C�ٚ@|(��vff@9��A�=qC��
                                    Bxm�*6  T          @ƸR@z=q�5�@z=qB�C�
=@z=q�s33@>�RA��C��                                    Bxm�8�  �          @�{@w
=�5�@{�B�C��@w
=�s33@@  A�C��)                                    Bxm�G�  �          @�
=@z�H�2�\@{�B��C�H�@z�H�qG�@@��A�p�C�
                                    Bxm�V(  T          @�\)@w��5�@~{B��C��@w��s�
@B�\A�{C��
                                    Bxm�d�  �          @ƸR@w
=�8��@z�HB\)C���@w
=�vff@>{A�RC���                                    Bxm�st  
�          @�\)@w��9��@z�HB33C���@w��w
=@>�RA��C��=                                    Bxm��  �          @�
=@y���6ff@z=qB��C��3@y���s�
@>�RA�\)C���                                    Bxm���  �          @�{@z=q�0��@{�B��C�h�@z=q�n�R@A�A�(�C�1�                                    Bxm��f  T          @�@w��/\)@~{B�RC�aH@w��mp�@Dz�A�=qC�R                                    Bxm��  �          @�
=@~{�3�
@xQ�Bp�C�]q@~{�p��@>{A�{C�G�                                    Bxm���  T          @ƸR@�G��6ff@p  B��C�` @�G��qG�@5�A��C��                                     Bxm��X  T          @��@|���:=q@n{B�C��{@|���s�
@2�\AׅC��                                    Bxm���  T          @�p�@�Q��9��@k�B�RC�3@�Q��r�\@0��AԸRC�P�                                    Bxm��  �          @�p�@����9��@hQ�BQ�C�4{@����qG�@-p�AЏ\C���                                    Bxm��J  �          @���@����<(�@g
=B�
C��@����s�
@+�A��HC�K�                                    Bxm��  �          @��
@����:=q@c�
B��C��@����qG�@)��A�
=C�u�                                    Bxm��  T          @Å@��\�:�H@_\)BffC�7
@��\�p  @%�A�p�C��3                                    Bxm�#<  
Z          @Å@��
�C33@Tz�B�C��)@��
�u�@Q�A�
=C���                                    Bxm�1�  "          @�{@w��C�
@S33B33C��{@w��u@
=A�33C��                                     Bxm�@�  T          @�z�@u��G
=@Mp�B�C�o\@u��w
=@  A�(�C�`                                     Bxm�O.  "          @�(�@{��J�H@@  A��C��
@{��vff@�\A���C�˅                                    Bxm�]�  �          @�ff@vff�N�R@-p�A�ffC�  @vff�u?޸RA�(�C���                                    Bxm�lz  �          @���@HQ��1�@N{B\)C�:�@HQ��b�\@
=A�z�C��                                    Bxm�{   T          @��@C33���@^{B&ffC��R@C33�R�\@,(�A�C�g�                                    Bxm���  T          @��
@8���33@c�
B/C��)@8���J=q@3�
B=qC�/\                                    Bxm��l  T          @�33@+��
=@r�\B@�C��f@+��B�\@EB\)C��\                                    Bxm��  
�          @�33@ �׿�p�@}p�BL=qC��3@ ���=p�@R�\B!  C�)                                    Bxm���  "          @��@
=��=q@�z�BX�RC��@
=�7�@`��B-p�C���                                    Bxm��^  �          @�p�@2�\���H@�ffBK�RC�~�@2�\�@  @a�B#  C�y�                                    Bxm��  T          @��@ff��  @��B��HC��@ff�*�H@���BZ�HC���                                    Bxm��  r          @�=q@)����z�@�ffBl  C��=@)���>�R@��\BC��C�˅                                    Bxm��P  �          @�z�@)����  @��\Bq
=C�J=@)���6ff@��BJ�C�xR                                    Bxm���  �          @��
@*�H��  @���Bp33C�W
@*�H�5@�
=BI�HC��3                                    Bxm��  �          @��@,(���(�@��HBp�C���@,(��5�@���BJ�RC�Ф                                    Bxm�B  T          @ƸR@-p����\@�{Bt�C�s3@-p��*=q@�p�BQG�C��=                                    Bxm�*�  T          @ə�@*=q��G�@�=qBxz�C�K�@*=q�+�@�G�BT��C�aH                                    Bxm�9�  �          @�@*�H��
=@�ffBwffC�3@*�H�$z�@�ffBT�
C�
=                                    Bxm�H4  �          @���@,(�����@���Bu��C��@,(��$z�@���BS(�C�&f                                    Bxm�V�  
(          @�@6ff��ff@��\BnG�C��
@6ff�)��@���BK��C��                                    Bxm�e�  �          @��@AG���@��BhQ�C�s3@AG��1G�@���BF{C�                                    Bxm�t&  T          @��@B�\���@��
Bh��C�!H@B�\�,(�@��\BG��C�=q                                    Bxm���  T          @ȣ�@?\)��=q@��HBi��C�\@?\)�*�H@��BH��C�                                      Bxm��r  �          @�  @>�R��ff@��HBj��C�7
@>�R�)��@�=qBI�RC�0�                                    Bxm��  "          @��H@B�\���
@��Bj�C��
@B�\�(��@�z�BJ�C��H                                    Bxm���  T          @�33@=p����R@�\)BnffC�� @=p��'�@�
=BM��C�>�                                    Bxm��d  T          @��@9����33@�33Bsz�C�&f@9���$z�@��BSQ�C�=q                                    Bxm��
  
�          @�p�@9����\)@��Bt
=C�n@9���"�\@�(�BT\)C�h�                                    Bxm�ڰ  "          @�
=@2�\���R@�p�Br\)C�
=@2�\�&ff@�p�BP�C��f                                    Bxm��V  �          @�  @333��  @�{Br=qC��3@333�'�@�BP�C�z�                                    Bxm���  T          @�=q@,(��c�
@���B}ffC��H@,(��z�@�\)B_=qC��f                                    Bxm��  T          @ƸR@.�R����@��RBvffC��@.�R�   @��BU�C���                                    Bxm�H  "          @ȣ�@1녿��R@��Bt
=C��)@1��'
=@�\)BRffC�c�                                    Bxm�#�  �          @���@.�R��\)@��BnffC���@.�R�=p�@��BH\)C�\)                                    Bxm�2�  �          @�G�@+���(�@��RBnp�C��
@+��0��@��BIC��                                    Bxm�A:  r          @��R@$z´p�@��
Bj�RC�
=@$z��+�@��\BE�C��                                    Bxm�O�            @�Q�@'����@�{Bl�C�g�@'��$z�@�BI��C�Ǯ                                    Bxm�^�  "          @��@#33���@�(�Br\)C��@#33�  @�ffBRz�C�E                                    Bxm�m,  "          @�  @z�z�@�ffB�z�C��R@z��=q@�z�Bf{C�޸                                    Bxm�{�  "          @���@�\�.{@�\)B���C��H@�\��
=@�z�BdC���                                    Bxm��x  T          @���@   �\(�@�(�Bv33C�|)@   ��@�  BX�\C��                                    Bxm��  
�          @�@p��(�@��
B��C���@p����H@���Bh{C��                                    Bxm���  
�          @�=q@p��
=@�Q�B��qC�C�@p���(�@�Bj�RC���                                    Bxm��j  
�          @�(�@#33�.{@���B��HC���@#33�z�@�p�BfQ�C��                                    Bxm��  
(          @���@#33�8Q�@���B�C�'�@#33�ff@�Bf
=C�:�                                    Bxm�Ӷ  
�          @�z�@�Ϳ�@��HB��C�xR@�Ϳ�(�@���Bl�\C��)                                    Bxm��\  "          @�33@�þ�Q�@�33B�aHC���@�ÿ�\@��\BsQ�C��\                                    Bxm��  
(          @��H@��\)@��B�ffC���@��Q�@��Bv��C�{                                    Bxm���  �          @�G�@�;�  @��B��fC��)@�Ϳ�@��
B{�C�t{                                    Bxm�N  
�          @�
=@(���{@���B��=C��3@(���p�@���Bx�C�˅                                    Bxm��  
�          @�{@�þ�@���B��3C��3@�ÿ�{@�\)Bv{C���                                    Bxm�+�  "          @���@����@��B��\C�u�@����@�  B{�HC��                                    Bxm�:@  �          @��R@ff��G�@��B��RC�  @ff����@���Bx�C���                                    Bxm�H�  
�          @��R@ff��
=@��B��)C�aH@ff��ff@���By33C���                                    Bxm�W�  T          @�(�@z�   @�\)B��{C�'�@z��\)@�Bv�C���                                    Bxm�f2  
�          @�(�@33��@��B��C��3@33��=q@�ffBx��C�.                                    Bxm�t�  "          @��?��H��@��
B��HC��3?��H��33@���B|�RC��                                    Bxm��~            @��R?�z����@�B�k�C�S3?�z῱�@��B�
=C��)                                    Bxm��$  
�          @��R?�\)?(�@�B�{A�(�?�\)�5@��B�33C�"�                                    Bxm���  "          @���?�(�?\(�@�Q�B��A�\)?�(����H@���B�k�C��\                                    Bxm��p  �          @�=q?��R?}p�@���B�z�B
=?��R��Q�@��B���C�/\                                    Bxm��  "          @��\?���?u@�Q�B��HA���?��;\@��\B��RC�N                                    Bxm�̼  "          @��?�?�ff@�  B�B�?���z�@��HB���C�3                                    Bxm��b  �          @��
?�G�?�
=@���B�#�B��?�G��.{@��B��3C���                                    Bxm��  
�          @��\?���?�ff@��B�B�B,�H?��׽L��@���B�L�C���                                    Bxm���  �          @�\)?�p�?O\)@���B��RB
=?�p���@���B���C�b�                                    Bxm�T  
�          @���?�\)?�@�  B��B!p�?�\)�8Q�@�(�B��C�U�                                    Bxm��  �          @�
=?��?�R@��\B�A�{?���8Q�@�=qB��fC��H                                    Bxm�$�  "          @��?��?+�@��\B�ǮB��?���+�@��\B���C���                                    Bxm�3F  T          @�
=?\(�?+�@��B�  B?\(��0��@��B�C��f                                    Bxm�A�  "          @�
=?�=q>\@��HB��A�Q�?�=q�s33@���B�\)C�Y�                                    Bxm�P�  �          @�  ?�\)?
=@��\B�W
A��?�\)�@  @��B�z�C�                                      Bxm�_8  �          @��\?u?��@�z�B�=qB?�?u���R@�\)B�.C�,�                                    Bxm�m�  
�          @��\?z�H?��
@��HB�ffBQ?z�H���
@��B���C���                                    Bxm�|�  
�          @�
=?z�H?��
@�\)B��RBRff?z�H�L��@�(�B�\)C�u�                                    Bxm��*  �          @�G�?u?��H@��\B��BN
=?u���@��RB���C��=                                    Bxm���  
�          @�{?c�
?�33@��
B��\Bv��?c�
>�=q@��B�33A���                                    Bxm��v  T          @���?s33?�=q@�Q�B�#�Bz�\?s33>�@���B�Q�A�                                      Bxm��  
�          @�p�?G�?��@��\B�B��?G�>�
=@�33B�  A��                                    Bxm���  �          @��?Y��?��H@�=qB��{B}�H?Y��>�{@��\B�Q�A�z�                                    Bxm��h  �          @��?��\?�ff@��B��Bq�
?��\>�(�@�z�B�A��
                                    Bxm��  
�          @���?��\@@�ffB~�HBk{?��\?O\)@��B�k�B��                                    Bxm��  T          @��\?�(�?��@��B���Bdp�?�(�?�@�p�B�.AǙ�                                    Bxm� Z  �          @��H?�G�?��@�z�B��
B[=q?�G�>��@�p�B�{A��
                                    Bxm�   
�          @��?���?�=q@�  B��3B\��?���>u@�\)B�ǮAH��                                    Bxm��  �          @��R?O\)?�p�@��RB�  Bt��?O\)=#�
@���B�G�@�R                                    Bxm�,L  �          @Å?n{?�ff@��\B�u�Bl
=?n{=��
@���B�\)@�(�                                    Bxm�:�  
�          @�33?�  ?��@�33B�k�BYG�?�  ��Q�@�Q�B��C�t{                                    Bxm�I�  
�          @�z�?��
?�Q�@��
B�aHBY�?��
�#�
@���B���C�ٚ                                    Bxm�X>  �          @ȣ�?��?�(�@��B�Q�B)z�?����\)@ÅB��RC��                                    Bxm�f�  "          @θR?��
?���@ƸRB�ffB,33?��
��Q�@�=qB�(�C�9�                                    Bxm�u�  
�          @ʏ\?��?�G�@ÅB��\BD?����=q@�\)B���C��f                                    Bxm��0  
�          @�Q�?h��?��\@�33B��3B@��?h�þ�G�@�p�B��\C�                                    Bxm���  �          @�\)?��\?�ff@�G�B��B7��?��\��ff@��
B�G�C�H                                    Bxm��|  �          @Ǯ?�=q?}p�@��B��
B*(�?�=q��@��
B��
C�                                      Bxm��"  �          @�z�?�{?Tz�@�\)B��Bz�?�{�&ff@�  B���C��R                                    Bxm���  �          @�p�?��?��@��RB�p�B3��?��<#�
@�(�B�L�?\)                                    Bxm��n  �          @�\)?�33?��@�z�B�{B/=q?�33��\)@���B��
C�xR                                    Bxm��  �          @���?�G�?B�\@�(�B���B��?�G��0��@�z�B�p�C��3                                    Bxm��  
�          @�=q?��H@\)@�{BG�Bc�R?��H?Y��@�=qB���A�G�                                    Bxm��`  T          @�=q?�Q�@z�@�=qB���Bp�?�Q�?&ff@��B�33A��                                    Bxm�  �          @�{?�=q?��@��B�.B[  ?�=q>�(�@���B��
A�G�                                    Bxm��  
�          @�p�?��?�@��HB��\Bf{?��>L��@�=qB���A(��                                    Bxm�%R  �          @�?#�
?�p�@�z�B�33B��?#�
>�  @�z�B�L�A�(�                                    Bxm�3�  
�          @�p�?:�H?���@�\)B�B�Bv��?:�H�\)@��
B�8RC�Z�                                    Bxm�B�  
�          @Å?.{?�33@��RB��3Bn
=?.{����@��B���C���                                    Bxm�QD  �          @��H?5?��@��RB��B^��?5��G�@���B�8RC�+�                                    Bxm�_�  
x          @��H?Y��?fff@�
=B���B9�H?Y���z�@�Q�B��=C��)                                    Bxm�n�            @�z�@p�?�33@���B~��BQ�@p�=�\)@��RB��?�
=                                    Bxm�}6  "          @�z�@{?��R@��B��
A��@{��Q�@�ffB���C�޸                                    Bxm���  
�          @�ff@�\?���@�(�B��RAǙ�@�\��=q@�\)B�\)C���                                    Bxm���  "          @��R@(�?aG�@��
B�z�A��R@(���ff@�p�B��HC�Ǯ                                    Bxm��(  �          @�  @�?L��@��RB��RA�G�@����@��B�.C���                                    Bxm���  
�          @���@�
?W
=@�  B�A�33@�
��@�G�B���C��=                                    Bxm��t  "          @�z�@'�?aG�@�\)B}
=A�  @'���@���B���C��3                                    Bxm��  �          @ƸR@�?:�H@��B���A��
@��(��@�p�B�
=C�b�                                    Bxm���  �          @���@   ?Tz�@��B�B�A�G�@   �
=q@��HB��
C���                                    Bxm��f  �          @ȣ�@ff?��H@�B�A�ff@ff�L��@��B��qC���                                    Bxm�  �          @�  @p�?��@�p�B��{B ��@p���@��HB��=C��=                                    Bxm��  �          @�@p�?�{@�33B�L�A�G�@p���@���B�{C���                                    Bxm�X  �          @ʏ\@!�?�=q@�=qBw=qA��@!�>B�\@�G�B���@�{                                    Bxm�,�  T          @ʏ\@#�
?��H@�33Bx�HA���@#�
=u@���B�.?��                                    Bxm�;�  �          @ə�@��?�\)@�z�B~=qA��@�ͽ#�
@���B�=qC��=                                    Bxm�JJ  "          @ə�@{?�\)@��B�=qB�\@{>L��@�z�B��=@��R                                    Bxm�X�  �          @�Q�@p�?У�@��
B�
B(�@p�>k�@�33B�z�@�(�                                    Bxm�g�  !          @ȣ�@?�z�@�=qB{  B��@>�=q@��B��H@�                                      Bxm�v<  �          @�Q�@�?��
@��Bz��B p�@�>\)@�Q�B��@Q�                                    Bxm���  
Z          @�G�?��R?�(�@�
=B���B#��?��R>�z�@��RB�ǮAz�                                    Bxm���  �          @ƸR?��?��@�ffB���B�?��=�@��B���@k�                                    Bxm��.  �          @�?�z�?У�@�(�B���B"33?�z�>aG�@��B��R@�(�                                    Bxm���  "          @�Q�?��H?��H@��\B��3B!��?��H<#�
@�Q�B�#�>�(�                                    Bxm��z  
�          @�(�?��
?���@�(�B���Bz�?��
��ff@ƸRB�ffC�^�                                    Bxm��   �          @�
=?�\)?�33@��RB��B p�?�\)���
@��B��)C�q�                                    Bxm���  
�          @�p�?���?�p�@��\B�W
BA\)?���<�@�Q�B�8R?�
=                                    Bxm��l  T          @�z�?��?�
=@���B��fBc��?��>k�@�G�B�
=A<��                                    Bxm��  �          @�ff?���?�\)@�{B��qBL?��ͽL��@�33B��3C��q                                    Bxm��  �          @���?u?ٙ�@��RB�B�Br�
?u>�=q@��RB�\A~�\                                    Bxm�^  �          @\?}p�?�33@�Q�B���Bk�
?}p�>W
=@��B��qA@Q�                                    Bxm�&  �          @�ff?��?޸R@��HB��Bk{?��>�{@�33B�� A��H                                    Bxm�4�  �          @��?��?�(�@�ffB�Q�Bi��?��>�33@��RB��)A��                                    Bxm�CP  T          @�ff?��?���@��
B�=qBb�?��>k�@��HB�ffAHQ�                                    Bxm�Q�  �          @�z�?xQ�?��\@�{B��B:33?xQ쾣�
@���B�\C��
                                    Bxm�`�  �          @�  ?c�
?c�
@�33B�ffB4(�?c�
��
=@��B�G�C�K�                                    Bxm�oB  T          @��?aG�?��\@�z�B��HBD��?aG����R@�\)B���C�O\                                    Bxm�}�  T          @�{?=p�?���@�Q�B�=qBcff?=p��8Q�@�(�B���C�N                                    Bxm���  �          @��>�\)?�G�@��B��HB�G�>�\)>aG�@�(�B�� B=q                                    Bxm��4  T          @�(��W
=?��@�=qB��B�녾W
=>�Q�@�=qB��B�
=                                    Bxm���  �          @�Q�#�
?���@�=qB��
B�k��#�
=u@�  B���B�G�                                    Bxm���  �          @�p�>�{?��\@�  B�B��>�{��Q�@�z�B��qC�b�                                    Bxm��&  T          @�G�=L��?�@�
=B���B�Ǯ=L��>�@�Q�B��qB�=q                                    Bxm���  "          @��H=�G�@.�R@�(�Bw��B�8R=�G�?�z�@�(�B��
B�                                    Bxm��r  �          @������?���@��B�\)B�  ����?!G�@��B�B���                                    Bxm��  �          @��׾�@�@��B��B�Ǯ��?8Q�@�ffB���B��q                                    Bxm��  �          @�z��@�
@�  B�  B�B���?+�@��\B�(�B�G�                                    Bxm�d  �          @���:�H?�(�@�Q�B�Q�B��:�H?�@��\B���C�R                                    Bxm�
  �          @�  �.{?���@�z�B�L�Bڊ=�.{>��@�B��qC}q                                    Bxm�-�  �          @ƸR���H?���@�\)B��B�L;��H=�G�@�{B���C'�H                                    Bxm�<V  T          @���?�z�@�33B�k�B�8R��>�(�@�z�B��3C�f                                    Bxm�J�  �          @�����?�{@��B��
B�k���>��@���B�ffC                                    Bxm�Y�  �          @����?�=q@�G�B�B�B���>�G�@�=qB���C5�                                    Bxm�hH  T          @����R?�
=@�\)B�L�Bמ���R?��@���B���C
�                                    Bxm�v�  T          @�{��
=@�@�=qB~=qB�q��
=?�G�@��B�CO\                                    Bxm���  �          @�Q쿴z�@)��@��Bj{B�녿�z�?���@���B��
C}q                                    Bxm��:  T          @�\)��(�?��@�z�B�{C5ÿ�(�=#�
@�=qB�B�C1�                                    Bxm���  "          @�
=��{?��
@�=qB��HB���{>�G�@��HB��
C�=                                    Bxm���  �          @��R��=q?�ff@�  B���B��f��=q>��@���B�z�C ��                                    Bxm��,  
Z          @�
=��?�z�@�Q�B��fB�B���?�@�=qB���C                                    Bxm���  T          @�Q쿢�\?���@��B�G�B�  ���\?   @�33B���CxR                                    Bxm��x  
�          @�����\)?޸R@�33B���C B���\)>Ǯ@��B�u�C$&f                                    Bxm��  T          @�G����?ٙ�@�ffBC
����>Ǯ@�ffB�\C(xR                                    Bxm���  �          @��ÿ�(�?�=q@���B�G�C	z��(�>�  @�Q�B��C+�{                                    Bxm�	j  �          @�=q�ٙ�?�
=@���B�u�CT{�ٙ�>�{@���B�L�C(��                                    Bxm�  T          @�=q���?ٙ�@��B�k�C	����>�p�@��B��{C(ٚ                                    Bxm�&�  T          @�녿�
=?���@�  B�ffC�׿�
=>u@�\)B���C,��                                    Bxm�5\  "          @�
=��z�?�33@�(�B{C+���z�>�33@�(�B�=qC)�=                                    Bxm�D  
�          @�
=����?�\)@�(�B~��CO\����>��
@��
B��{C*�                                    Bxm�R�  �          @�����H?�z�@�(�B}��C�{���H>�Q�@�(�B�Q�C)��                                    Bxm�aN  T          @�Q�� ��?�
=@�(�B{��C8R� ��>�p�@�(�B�W
C)u�                                    Bxm�o�  �          @�33�
=q?�\)@�Bz  C.�
=q>���@�p�B�(�C,                                      Bxm�~�  "          @�=q�  ?�p�@��By�C�)�  >8Q�@��
B�u�C/�=                                    Bxm��@  T          @����	��?���@�{B}C{�	��>\)@�(�B�\)C0aH                                    Bxm���  �          @�Q��	��?�
=@���B}�Cc��	��>�@��HB��C0��                                    Bxm���  T          @�
=�   ?�z�@�p�B�W
C���   =���@��B��C10�                                    Bxm��2  �          @�\)�{?��@�  Bs(�C��{=#�
@�p�B��HC2�R                                    Bxm���  	�          @�  �#�
?�@�Q�Br�Cs3�#�
����@���B~Q�C6!H                                    Bxm��~  �          @�G��'�?s33@�=qBt�
C 
�'����
@���B{ffC;�                                    Bxm��$  "          @�G��{?z�@��RB�C&�=�{�5@�{B~�\CD�                                    Bxm���  
�          @���Q�?
=q@��
B�p�C%��Q�J=q@��HB��
CH^�                                    Bxm�p  T          @�Q���H>��@���B�Q�C()���H�n{@��HB���CMG�                                    Bxm�  �          @��ÿ�\>�ff@�\)B�.C%���\�h��@�B��RCOB�                                    Bxm��  
�          @�{��=q?�\@�{B��C!�3��=q�fff@���B�W
CQ�q                                    Bxm�.b  T          @�Q쿣�
>��R@��B���C&�����
���@�Q�B��qC]��                                    Bxm�=  "          @�G����>W
=@�{B���C)!H�����G�@��B�  Ce\                                    Bxm�K�  T          @�  �L��=�Q�@�B��C-���L�Ϳ�\)@���B�8RCo��                                    Bxm�ZT  �          @��ÿ
==��
@��B��qC,LͿ
=��33@��\B���CwG�                                    Bxm�h�  T          @��ÿ0��=�Q�@��B�ǮC,�Ϳ0�׿��@�=qB�33Cs�f                                    Bxm�w�  �          @���=p�>W
=@��HB��C$\)�=p����R@��RB���Co�                                    Bxm��F            @��H��  >Ǯ@�p�B�  C"�{��  ���\@�33B��
C[=q                                    Bxm���  �          @���{���
@�{B�
=C4���{��{@���B�Q�C\J=                                    Bxm���  A          @��R��G��8Q�@�p�B���C9��G���G�@��RB��C\��                                    Bxm��8            @��׿��W
=@��\B��C<z῵��=q@�33B���Cd!H                                    Bxm���  �          @�  ����#�
@��\B�ffC:�
������
@��
B���Ce}q                                    Bxm�τ  �          @�ff����u@���B�k�C6}q�����
=@��B�B�Cc�                                    Bxm��*  �          @����{    @�p�B���C3ٚ��{���@�  B��\C[��                                    Bxm���  "          @���޸R=L��@�(�B��C2���޸R���@�\)B��CX��                                    Bxm��v  T          @������R>��@�{B��C/�R���R���H@��B���CS5�                                    Bxm�
  
�          @�=q��p�����@���B���CB�)��p���(�@�z�B�33Cju�                                    Bxm��  �          @��\�(�ÿ(�@�  B�.C^���(���G�@�p�B�#�C{�=                                    Bxm�'h  T          @����W
=�W
=@��RB��)CA�ͿW
=��\)@�\)B�33Cr�R                                    Bxm�6  T          @���\(���@��B��{C<
=�\(����
@�ffB�\)Cp�f                                    Bxm�D�  �          @�녿   ���R@�Q�B�L�CS�=�   ��p�@�  B���C}�{                                    Bxm�SZ  �          @�  >�=�@�
=B���B)�>���Q�@���B�#�C���                                    Bxm�b   
Z          @�z�O\)�@  @�G�B�33C^��O\)��\@���B�B�Cz�H                                    Bxm�p�  "          @�
=��ff���@�B��C^Y���ff����@�z�B�� C�s3                                    Bxm�L  �          @�{���=���@��B�z�C@ �������@��B��RC��3                                    Bxm���  "          @�=q?�\>L��@���B�#�A���?�\��  @��B��C��                                    Bxm���  T          @�p�>��u@���B��C���>�����@�{B���C�l�                                    Bxm��>  s          @�(�?\(���\)@��B���C�R?\(����@���B�(�C���                                    Bxm���  �          @�ff?s33��
=@ÅB���C��?s33��Q�@���B��C��                                    Bxm�Ȋ  T          @��?W
=��@�
=B�p�C�ٚ?W
=��@��
B���C�Ф                                    Bxm��0  	�          @�G�>�녾��H@�ffB�G�C��>����@�(�B�#�C��=                                    Bxm���  5          @ə��u��G�@ÅB��C�c׾u�@  @���Bu��C��=                                    Bxm��|  T          @�녿
=q��z�@�\)B�8RC~B��
=q�W
=@��BeQ�C�p�                                    Bxm�"  "          @ƸR=�\)��  @�{B��{C�]q=�\)�<��@��
BtC���                                    Bxm��  
Z          @�(�?Y���(��@���B���C��?Y���
=q@��B�W
C�Ǯ                                    Bxm� n  
�          @��
?�G��(��@�(�B���C�Ff?�G���@���B��
C�b�                                    Bxm�/  
Z          @ƸR?}p�����@��HB�u�C��?}p��QG�@�Bbz�C�s3                                    Bxm�=�  
�          @��?^�R�!�@��B�\)C�l�?^�R�x��@��HBJ�C�Ff                                    Bxm�L`  	�          @�?���6ff@��\Br\)C���?����p�@�B;�\C���                                    Bxm�[  �          @��H?����/\)@��Bx�C��=?������\@�\)BA  C��=                                    Bxm�i�  
(          @�G�?���)��@�p�B�ǮC�W
?����  @��BG��C�=q                                    Bxm�xR  �          @��H?5�-p�@�B{C�K�?5��=q@��BEffC��                                    Bxm���  �          @�(�?fff�*=q@�
=Bz�C�N?fff����@��BF�C�C�                                    Bxm���  �          @���?
=�5�@�ffB|Q�C���?
=��{@�G�BB{C��                                    Bxm��D  "          @�{>���<(�@�ffBz�C��>����G�@���B?ffC��                                     Bxm���  "          @�G�?@  �@  @�Q�BwG�C��q?@  ���
@��B=Q�C��R                                    Bxm���  �          @�Q�?(��H��@��Bq��C���?(����@�p�B7z�C��f                                    Bxm��6  
�          @�ff=����G�@�z�Bt=qC���=������R@��B8��C���                                    Bxm���  
�          @���\)�<(�@��Bxp�C��R�\)����@�
=B=��C�Ff                                    Bxm��  "          @�zᾊ=q�>�R@�z�Bx{C�p���=q���\@�{B<��C�<)                                    Bxm��(  �          @�{�k��AG�@�Bw�\C��R�k���(�@��RB<(�C���                                    Bxm�
�  �          @�=q��\)�HQ�@���Bv{C�P���\)��Q�@���B:�C��f                                    Bxm�t  T          @��#�
�4z�@��B��C��ý#�
����@�(�BH��C���                                    Bxm�(  �          @�{�#�
�.�R@��HB���C�E�#�
���R@�{BK�C��H                                    Bxm�6�  
�          @��
�Ǯ��R@�\)B�p�C�  �Ǯ�u�@�B_��C��                                    Bxm�Ef  
�          @�\)>�
=�R�\@�33Bq�
C���>�
=��{@���B6Q�C��                                    Bxm�T  "          @��>��L��@�\)Bvp�C�E>���z�@�ffB:��C���                                    Bxm�b�  �          @�33?���8Q�@�{B�L�C�^�?����(�@�\)BGG�C���                                    Bxm�qX  A          @�=q    �3�
@ƸRB�\)C�f    ��=q@���BJ��C��                                    Bxm��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��J              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��<              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��.              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�!               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�/�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�>l              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�M              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�j^              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�y              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��P              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��B              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�ߎ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��4              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�&              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�(�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�7r              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�F              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�T�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�cd              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�r
              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��V              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��H              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�ؔ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�,              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�!�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�0x              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�?              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�M�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�\j              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�k              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�y�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��\              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��N              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�њ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��@              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�2              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�)~              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�8$              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�F�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�Up              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�d              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�r�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��b              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��T              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�ʠ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��F              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�8              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�"�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�1*              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�?�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�Nv              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�]              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�k�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�zh              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm�æ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxm��L  F          A   @�Q�@.�R@�=qB:A�\@�Q�?\(�@��
BS�A$z�                                    Bxm���  
�          @���@��\@.�R@��
B5z�A�(�@��\?n{@�BN  A-��                                   Bxm��  
�          @�@�33@H��@�B,�HB@�33?�\)@��
BJ33A|                                      Bxm��>  
�          @�\)@��\@Z=q@��B��B��@��\?�(�@���B?
=A�p�                                    Bxm��  T          @��@�Q�@\(�@�p�B33B�R@�Q�?�ff@�\)B7ffA�
=                                    Bxm��  �          @��R@��R@�@�
=B$�RA�{@��R?=p�@��RB8=q@��                                    Bxm�*0  "          A z�@���@p�@�
=B(�A�@���?W
=@�\)B,ffA�
                                    Bxm�8�  �          Ap�@��
@X��@�\)B�A��H@��
?�ff@�G�B,�A�{                                    Bxm�G|  "          Aff@�ff@vff@��HB  B33@�ff@ff@���B"��A���                                    Bxm�V"  
�          A�R@���@��@|(�A���B�@���@5@�G�Bp�A�p�                                    Bxm�d�  
�          A�@���@�(�@�=qA�
=B{@���@;�@��RB
=Aۅ                                    Bxm�sn  �          A�\@���@�Q�@��RA�ffB33@���@1�@��B"G�A���                                    Bxm��  
�          A�H@��\@w
=@�z�A��HB�@��\@=q@��
B(�A���                                    Bxm���  �          A@���@~{@��BG�B\)@���@��@��
B%��A�ff                                    Bxm��`  �          A�@�\)@���@|(�A�=qB�@�\)@>�R@�33B
=A�                                      Bxm��  
�          AG�@�
=@��@��HB�BG�@�
=@!�@��
B&�HAƣ�                                    Bxm���  "          A=q@���@qG�@��
B
p�B	Q�@���@
�H@��B,\)A���                                    Bxm��R  "          A@���@mp�@���B�BG�@���?�Q�@�G�B@��A�z�                                    Bxm���  �          A@�  @S33@�
=B)�B��@�  ?���@�
=BH=qA�                                    Bxm��  
(          A=q@�@Q�@��\B-33B\)@�?��@�=qBK�
A|                                      Bxm��D  T          A�@��@U@�33B%
=B�\@��?�G�@��
BC�
A�{                                    Bxm��  
�          A(�@���@G�@�  B0\)A�(�@���?�Q�@�BL��AS�                                    Bxm��  4          A=q@��R@b�\@�G�B%\)B�H@��R?��@��
BE�A�(�                                    Bxm�#6  "          AQ�@��@#�
@���B%��A��@��?@  @�B:Q�@�
=                                    Bxm�1�  
�          A(�@���@S33@���B�RA�33@���?�p�@��B=
=Au�                                    Bxm�@�  �          AQ�@���@I��@�G�B{A��@���?��@�Q�B:p�AZ=q                                    Bxm�O(  "          A(�@���@"�\@��B�HA�\)@���?@  @��HB3@�z�                                    Bxm�]�  �          A  @�=q?�(�@���B3=qA��@�=q�8Q�@�G�B==qC��                                    Bxm�lt  
�          A��@��H@p�@���B33A�{@��H?@  @�=qB(�H@��                                    Bxm�{  T          A��@���@'
=@��HB z�A�ff@���?J=q@�z�B5(�@��\                                    Bxm���  
�          A��@��R@(�@�33B3=qA��H@��R>u@ǮBC{@{                                    Bxm��f  	�          A@��?��@�33BO�Ap��@�녿G�@�BSffC�g�                                    Bxm��  T          A�@���?k�@��
B\p�A8��@��ÿ�(�@ҏ\BZ{C�u�                                    Bxm���  
�          A=q@�z�?�p�@ȣ�BI�A�(�@�zᾳ33@�  BS�C���                                    Bxm��X  "          A�R@���@!G�@��\B"�A�{@���?333@�33B7�@�Q�                                    Bxm���  
�          A�R@�33@
=@��
B\)A�(�@�33?(�@��B.��@�
=                                    Bxm��  
�          A(�@�{?�G�@�{BGz�AxQ�@�{�333@�=qBL�C�Q�                                    Bxm��J  
Z          A=q@�ff?��
@���BA�Ap  @�ff�.{@�G�BG(�C��q                                    Bxm���  �          A
�\@��\?�(�@�{B@z�AlQ�@��\�+�@�=qBE�
C���                                    Bxm��  
Z          A
{@���?���@��HB=\)A���@������R@��HBGQ�C�e                                    Bxm�<  
�          AQ�@��H?��H@���B>�RAC33@��H�^�R@ʏ\B@��C���                                    Bxm�*�  
z          A	��@���?#�
@ʏ\B=��@��
@��ÿ�@�ffB8��C�3                                    Bxm�9�  �          A�@��?\)@�
=B6\)@��@�녿�(�@�=qB0��C�/\                                    Bxm�H.  S          A\)@��H���
@��BG�C���@��H��R@ȣ�B8��C�#�                                    Bxm�V�  
�          A
=q@�(��p��@��BAG�C�H�@�(��=p�@�Q�B(�RC�#�                                    Bxm�ez  �          A��@������@�p�BE�\C�K�@���N�R@�{B*G�C�                                    Bxm�t   �          AQ�@�{�=p�@�\BY�RC���@�{�E�@�{B?�
C���                                    Bxm���  �          AQ�@�  ����@�BZ��C��@�  �p��@ϮB833C���                                    Bxm��l  "          Ap�@�����@�ffBO��C��H@�����\)@���B'�\C��3                                    Bxm��  �          A
ff@�ff����@ᙚBY��C�q�@�ff����@�=qB1��C��                                     Bxm���  �          A33@��\)@��BLQ�C��@����@�z�B�
C�3                                    Bxm��^  
�          A�@��I��@�G�B@z�C�@���G�@��B(�C�p�                                    Bxm��  �          Aff@����J�H@ʏ\BCQ�C��H@�����=q@�B�
C���                                    Bxm�ڪ  �          AQ�@����E�@���BH33C��R@������@���B�C��                                    Bxm��P  "          Az�@�G��{@�33BK33C�Y�@�G���Q�@�p�B
=C�Z�                                    Bxm���  
�          A	G�@��R�#�
@�ffBMQ�C���@��R��(�@�\)B�C��)                                    Bxm��  �          A	G�@�p��'�@ָRBM�C�Y�@�p���{@��RB{C�p�                                    Bxm�B  �          A	��@����*=q@���BJ\)C�ff@������R@�z�B�C���                                    Bxm�#�  T          A	p�@�33��H@ڏ\BS
=C��@�33����@�z�B$ffC���                                    Bxm�2�            A	�@���%�@��BW(�C��{@����\)@��B%�C�U�                                    Bxm�A4  �          A	G�@�ff�+�@ڏ\BR��C�q�@�ff����@�G�B!
=C�e                                    Bxm�O�  �          A	p�@�=q�=p�@ٙ�BQ�C���@�=q����@�p�B�RC�E                                    Bxm�^�  �          A{@��
�p�@�Bi  C��=@��
���@�(�B7�C��=                                    Bxm�m&  	.          A��@�녿�=qA (�By\)C���@���l��@�{BRp�C��                                    Bxm�{�  �          A=q@w���
=@��RBC��@w��Dz�@�B`�\C��\                                    Bxm��r  �          A�@mp���\)@�B~  C��f@mp��|��@�Q�BQ(�C���                                    Bxm��  �          A�
@j=q�ٙ�A z�B}33C��=@j=q���@߮BL�C�0�                                    Bxm���  �          AG�@Z=q��z�A ��B��C��)@Z=q�s�
@�BZ(�C��3                                    Bxm��d  �          A33@Z=q��G�@�p�B��=C��@Z=q�'
=@�{Bh��C�K�                                    Bxm��
  �          @�
=@Tz�?��@�RB���A#�
@Tz���@߮BuffC�G�                                    Bxm�Ӱ  �          @�z�@G�?��@�ffB��
A-�@G����@׮Bw��C�\                                    Bxm��V  �          @�Q�@�>k�@�B�\@���@����@أ�B���C���                                    Bxm���  �          @�Q�@p�?�@�z�B���Ag�@p���33@��B�W
C���                                    Bxm���  �          @�\@#�
>�\)@��
B�Q�@�{@#�
�
�H@�G�B~�C��
                                    Bxm�H  T          @���@7
=?   @�RB�.A   @7
=���R@�ffB}�\C���                                    Bxm��  �          @�G�@8��>���@�{B��{@љ�@8���z�@�z�Bwp�C�7
                                    Bxm�+�  �          @�  @~{@+�@��BN��BG�@~{>�G�@ӅBk��@�33                                    Bxm�::  �          @�(�@�p�@G
=@��BB�\B�H@�p�?c�
@��
Bd��AA�                                    Bxm�H�  �          @�(�@�ff@L(�@��B@{B�H@�ff?z�H@�33BcffAQ��                                    Bxm�W�  �          @��R@�=q@p  @���B0  B#��@�=q?�{@�
=BZ33A��                                    Bxm�f,  T          A z�@��@AG�@��BH�RB�
@��?333@�=qBi(�A��                                    Bxm�t�  �          @�ff@���@.�R@���BQ{B�
@���>Ǯ@ڏ\Bm@�G�                                    Bxm��x  �          @��@vff@ff@ӅB^A�G�@vff��Q�@�Q�Bt��C�Z�                                    Bxm��  �          A ��@|��@��@�(�B^{A�=q@|�;8Q�@�  Br(�C��\                                    Bxm���  �          A{@��@%@љ�BV�RB33@��>��@�G�Bp�@
=                                    Bxm��j  T          A@�G�@A�@�{BF�RB�@�G�?.{@ۅBf�\A��                                    Bxm��  �          A�@qG�@h��@���BF=qB0{@qG�?��\@���Br{A�z�                                    Bxm�̶  �          A  @w�@�
=@�G�B5G�B=�
@w�?�
=@��Bg�RAӮ                                    Bxm��\  T          AQ�@�=q��R@�  BI33C���@�=q�<��@�33B0\)C�                                    Bxm��  �          A\)@�  ��\)@�
=BJ�RC���@�  ��H@ȣ�B8��C�                                      Bxm���  �          A�@�
=?^�R@�p�BM\)A��@�
=��\)@�G�BG�
C��H                                    Bxm�N  �          A�@�Q���@�(�BM(�C�8R@�Q��#�
@�(�B9��C��f                                    Bxm��  �          A(�@��
�L��@��BP��C���@��
�(Q�@�z�B;C���                                    Bxm�$�  �          A��@�ff�L��@أ�BQ�HC��3@�ff���@�=qB>C�h�                                    Bxm�3@  �          A  @���L��@�  BRG�C��q@����@ə�B?(�C�aH                                    Bxm�A�  �          A��@�(����
@�33BT��C�޸@�(����@�z�BAp�C�:�                                    Bxm�P�  �          A	G�@�
=?8Q�@�Q�BP33@���@�
=��p�@�=qBH\)C���                                    Bxm�_2  �          A	�@�p�?��
@�\)BO�A3�
@�p���Q�@���BLp�C�<)                                    Bxm�m�  �          A��@��R?�\)@�BN��Ab=q@��R��  @�(�BO�C���                                    Bxm�|~  �          A�@��R?�33@��
BGA���@��R�.{@�33BP�C�p�                                    Bxm��$  T          A	�@���@(�@�\)BDA�  @��;�\)@ڏ\BS��C�t{                                    Bxm���  
�          A	��@�{@
�H@θRBC�
A�\)@�{��z�@��BR�\C�aH                                    Bxm��p  �          A��@�z�?�G�@��HBJ�HAz=q@�zῆff@�BNffC�z�                                    Bxm��  �          A�@�=q@
=@�
=BR�A�33@�=q��@�(�Bg�C�'�                                    Bxm�ż  �          Aff@�z�?��@�p�BP(�A���@�z�\)@��B[�
C���                                    Bxm��b  �          A ��@�p�?�Q�@���BL�A���@�p���@��BY�RC�0�                                    Bxm��  �          A ��@���@�@ƸRBI�RA�
=@��;aG�@ҏ\BZ��C���                                    Bxm��  �          A z�@��
?�Q�@�Q�BM=qA��
@��
��@�G�BZ\)C�!H                                    Bxm� T  �          @�\)@�  @�H@�{BN��A�33@�  <#�
@���Be��=�\)                                    Bxm��  "          A{@aG�@��\@���B:ffBK�@aG�?��@�=qBr(�A�p�                                    Bxm��  T          A ��@^{@��R@���B-z�BV�@^{@�@��
Bj�B	��                                    Bxm�,F  �          A ��@P  @�p�@�
=B+�BbG�@P  @%�@�z�Bk�
B�                                    Bxm�:�  �          A�@#�
@��@��B)��B�Q�@#�
@A�@��HBr��BG{                                    Bxm�I�  
�          A
=@9��@���@��B!�By�@9��@L(�@޸RBh��B>�H                                    Bxm�X8  �          A
=@+�@ȣ�@�
=BG�B�Ǯ@+�@�Q�@�p�B\��Ba�                                    Bxm�f�  �          A=q@.�R@ȣ�@��B�B�@.�R@���@�=qBY�RB`(�                                    Bxm�u�  �          A=q@0  @��
@�{B=qB�Q�@0  @�ff@ָRBT�HBc{                                    Bxm��*  �          A{@AG�@\@��
B�RB~33@AG�@vff@أ�BX��BOp�                                    Bxm���  �          A=q@@��@���@�G�A���B��\@@��@�\)@���BGz�B`��                                    Bxm��v  �          A�@QG�@ۅ@W�A�p�B��@QG�@��
@�p�B,(�Be��                                    Bxm��  T          A�
@:�H@�  @��A�B�=q@:�H@�Q�@���BAz�Bi��                                    Bxm���  �          A\)@'
=@�=q@��A��B�{@'
=@�  @�
=BH{Bt�
                                    Bxm��h  �          A�H@'
=@���@`��A˅B���@'
=@�33@�=qB6  B{�R                                    Bxm��  �          A{@  @�G�@��
BG�B��@  @~�R@���Bl�Br(�                                    Bxm��  �          A?�{@�(�@�ffB!�B�  ?�{@���@���Bv��B�(�                                    Bxm��Z  �          A�\?aG�@���@�  B)Q�B���?aG�@s�
A (�B�B�                                    Bxm�   T          A=q?���@��H@�(�B.ffB�\)?���@e�A ��B�#�B��                                    Bxm��  �          A��?��R@�  @�z�B.ffB�33?��R@Vff@��RB�(�B�.                                    Bxm�%L  �          A	��?���@�  @�\)B'{B��H?���@h��@��B{B��                                    Bxm�3�  �          A?У�@��@�{B?ffB�.?У�@(Q�@�  B�u�Bi                                      Bxm�B�  �          A�R?�@��@�\)B?
=B�?�@'
=@�G�B���B]\)                                    Bxm�Q>  �          Ap�?��
@�  @�Q�B8�\B�8R?��
@4z�@�z�B��Bg�                                    Bxm�_�  �          A��?�ff@���@�p�B6{B��f?�ff@7�@�=qB��qBg��                                    Bxm�n�  �          A?�(�@��@��B2G�B�B�?�(�@>{@�B�u�Ba�\                                    Bxm�}0  �          AG�?�Q�@�p�@�G�B/�HB�(�?�Q�@C33@�Q�B��\Be�H                                    Bxm���  �          A  ?�Q�@��@�p�B-�HB�#�?�Q�@Dz�@���BQ�Bf�H                                    Bxm��|  �          A(�?�p�@�(�@�Q�B1��B��)?�p�@@  @�
=B�B�Bpff                                    Bxm��"  �          AQ�?�G�@�33@���B)
=B�z�?�G�@R�\@��
B|��Bw33                                    Bxm���  �          A��@'�@\@�  B(�B�ff@'�@l��@�{Bc\)BZ��                                    Bxm��n  �          A(�@#�
@�p�@��B�RB�G�@#�
@_\)@�G�Bi�BW{                                    Bxm��  �          A  @Q�@�z�@��\B!z�B�.@Q�@Y��@�BsG�Bgff                                    Bxm��  �          A�R?���@�G�@�{B'\)B�u�?���@P  @�  Bz��Bq�                                    Bxm��`  �          A=q?�\)@�33@�B1
=B���?�\)@>�R@���B�ǮBv�                                    Bxm�  �          A�\@  @���@��B0=qB���@  @2�\@陚BG�BLff                                    Bxm��  �          A33@1G�@�=q@��HB,
=By��@1G�@/\)@�RBvp�B2��                                    Bxm�R  �          A{@@���@��B9�\B�(�@@!G�@�(�B�� B<ff                                    Bxm�,�  �          A�?�=q@�p�@��\B1ffB�.?�=q@>{@�=qB��=Bi�                                    Bxm�;�  �          A�\@ff@�\)@�Q�B6��B��@ff@-p�@��B���BQ33                                    Bxm�JD  �          A�R@@��@�
=B+ffB�z�@@G�@���B~
=B`��                                    Bxm�X�  �          A{@�
@���@�z�B �RB�p�@�
@^{@�=qBt�RBm=q                                    Bxm�g�  �          A
=?Ǯ@��@�33B9\)B�?Ǯ@1�@��B��{Bs33                                    Bxm�v6  �          A�R?�=q@���@��B+B�?�=q@S33@�z�B�\B��                                    Bxm���  
�          A33?�@�  @���B,\)B�W
?�@P��@��B�.B��f                                    Bxm���  �          A�?�ff@�@���B:(�B�Q�?�ff@3�
@�z�B��B��\                                    Bxm��(  �          A=q?���@�\)@���B7��B��?���@9��@�G�B��qB���                                    Bxm���  �          A\)?333@�Q�@�(�BC��B�� ?333@#33A z�B��{B�B�                                    Bxm��t  �          A��?E�@��@ə�B=�B���?E�@2�\A ��B�� B�L�                                    Bxm��  �          A
{?aG�@�ff@�B@p�B�z�?aG�@,��A�\B���B��f                                    Bxm���  �          A33?��\@�\)@θRB?B���?��\@-p�A33B�\B��3                                    Bxm��f  �          A	�?Y��@�33@ϮBC�B���?Y��@$z�A�RB�aHB��\                                    Bxm��  �          A�?��@�@�=qB0{B���?��@P��A Q�B��\B�                                    Bxm��  �          A��?�{@�z�@�Q�B p�B�
=?�{@k�@��HBz��B��                                     Bxm�X  �          A�H?�Q�@�G�@�
=B"(�B���?�Q�@e@���B}\)B�u�                                    Bxm�%�  �          A33?8Q�@�33@���B#33B�\?8Q�@g�@�33B��B��                                    Bxm�4�  �          A�?�\@�
=@�B�\B���?�\@p��@��B}G�B��                                    Bxm�CJ  
�          A�?�R@��
@�  B\)B�L�?�R@}p�@�RBw{B�8R                                    Bxm�Q�  �          A  >�G�@��@�B��B��q>�G�@�33@�G�Bl=qB��                                     Bxm�`�  �          A(�>�@�p�@���BG�B�ff>�@��@ᙚB_�B��=                                    Bxm�o<  �          A��>�33@�
=@��\B\)B�u�>�33@���@��HB`{B���                                    Bxm�}�  �          A
{?z�@�@�33BB���?z�@��@陚BgQ�B�k�                                    Bxm���  T          A33?k�@�ff@���B��B�k�?k�@���@�G�Bc�
B���                                    Bxm��.  �          A�?��@�ff@���B(�B�Ǯ?��@���@���Bb�B���                                    Bxm���  
�          Az�>�@�p�@��B (�B��>�@���@�Q�B_ffB�L�                                    Bxm��z  S          A	>�@�(�@��A�=qB��\>�@��R@��BZ�B���                                    Bxm��   D          A
�\>��H@��H@�=qA�
=B�L�>��H@�33@��B_
=B�\)                                    Bxm���  d          A
=>�p�@�  @�  B�
B��>�p�@�{@�G�Bd�B��f                                    Bxm��l  �          A\)>aG�@���@�  Bz�B�� >aG�@�
=@��Bd��B��{                                    Bxm��  T          A
�R?=p�@�33@��B
(�B�(�?=p�@�\)@�(�Bi�RB�Q�                                    Bxm��  
�          A33?fff@�  @�{A��B��f?fff@���@���BVB�G�                                    Bxm�^  �          A�\?#�
@�@��\B1p�B��R?#�
@AG�@��B��B�(�                                    Bxm�  �          A��?!G�@�=q@�Q�B@B�{?!G�@ ��A   B��B���                                    Bxm�-�  �          A\)?��@�{@ʏ\B-=qB��?��@`  AQ�B��qB�ff                                    Bxm�<P  �          A��?E�@�=q@޸RB9B��\?E�@HQ�A��B�ǮB�.                                    Bxm�J�  �          A?��R@��R@�Q�B_{B�?��R?�
=A=qB�
=BDG�                                    Bxm�Y�  �          Ap�?z�H@�{@���BJ\)B�z�?z�H@�\A  B�B�B���                                    Bxm�hB  �          A33?n{@��R@��B[=qB���?n{?�p�A�RB�u�Bv��                                    Bxm�v�  �          A�?��@��A{B_z�B�Ǯ?��?ǮA=qB��)B^�H                                    Bxm���  �          Aff?fff@���@��HBY�
B�z�?fff?��
A�HB�{B|�                                    Bxm��4  �          A��?z�H@��R@�\)BU�RB���?z�H?�{A�B��)By                                      Bxm���  �          AG�?���@��H@�=qBXB�#�?���?��HA��B���Bbp�                                    Bxm���  �          A�?��R@��@��RBY�B�\?��R?�A�
B��3BU                                    Bxm��&  �          A�
?�Q�@�
=@�z�BW{B�u�?�Q�?��
A�B���Ba=q                                    Bxm���  �          A�?�Q�@�=qA ��B_�\B��q?�Q�?�(�Az�B��=BL
=                                    Bxm��r  �          A��?��
@��H@��
BT{B�.?��
?��A(�B�B�B_�                                    Bxm��  �          A�\?�@�z�@�z�BOQ�B��3?�@G�AG�B��=B[�                                    Bxm���  �          A
=?��H@��@��HBJ�B�{?��H@
=qA��B�\B_�
                                    Bxm�	d  �          A
=?�
=@�\)@ᙚBH=qB�
=?�
=@�RA��B�z�Be�H                                    Bxm�
  �          A��?���@���@�
=BJ=qB���?���@p�A�B��
Bi�
                                    Bxm�&�  �          A�H?���@���@�\)BH(�B��3?���@33A��B���Bl\)                                    Bxm�5V  �          A33?�
=@��@�BO\)B�?�
=@   A{B��fBZ                                      Bxm�C�  �          A�?��@�z�@�p�BKz�B�B�?��@��A\)B�Bj(�                                    Bxm�R�  �          A��?�33@���@�
=BI��B��?�33@(�A  B��qBf�                                    Bxm�aH  �          A�
?��@�ff@߮BD��B��
?��@�HA��B��RB��H                                    Bxm�o�  �          A
=?.{@\@���B?��B�33?.{@(Q�A�B�aHB��f                                    Bxm�~�  T          A  ?�  @��@��BT\)B��3?�  ?ٙ�A�
B���BB33                                    Bxm��:  �          A��?�{@��@�\)B]G�B�p�?�{?�=qA=qB���B1G�                                    Bxm���  �          A{?�G�@�(�@�Ba��B��?�G�?�A\)B�W
B+�                                    Bxm���  �          A�\?���@��@�33B`B�Ǯ?���?�33A
�RB�(�B#�\                                    Bxm��,  �          A�?�  @�  @�Bg�B�.?�  ?W
=A  B��
A�33                                    Bxm���  �          A��@@���@���BL
=B���@?�\A	�B�B!ff                                    Bxm��x  �          A�@p�@�33@ۅBB��B��)@p�@
=A	��B���B.                                    Bxm��  �          A?�  @���@߮BH�HB�{?�  @ ��A
=B��BD�                                    Bxm���  �          A�R?���@��@�
=BO��B�Ǯ?���?�=qA	��B�\Bl�                                    Bxm�j  �          A�
��G�@A�A��B��B���G��\(�Ap�B��C\�                                    Bxm�  �          A��ٙ�@X��Ap�B��
B�B��ٙ���G�A(�B�aHCBxR                                    Bxm��  �          A=q�:�H@^�RA=qB�ǮB���:�H�\A��B�W
CO.                                    Bxm�.\  �          A{��ff@H��A�
B�L�B��H��ff�=p�A��B�k�CWE                                    Bxm�=  T          Aff�xQ�@L(�A(�B�Bսq�xQ�333A�B��{CW�3                                    Bxm�K�  �          A�(��@Mp�A�B��B�LͿ(�ÿ.{A��B��qCa��                                    Bxm�ZN  �          A\)�+�@QG�A�B�L�B�\�+��+�A�\B��HCa8R                                    Bxm�h�  �          A33��@`��A33B�=qB�uþ����A�\B���C\c�                                    Bxm�w�  �          A��?�  @ʏ\@���B'�\B�
=?�  @HQ�A�B�Bs                                      Bxm@  �          A�
@fff@��H@eA��RB���@fff@��@���B5�Bb�H                                    Bxm�  �          A@���A�
?˅A��Br\)@���@�
=@���A��B]Q�                                    Bxm£�  �          A�\@���A ��?�ffA0z�Br@���@�ff@�z�B�B[��                                    Bxm²2  �          A@��A�@  AX��BwG�@��@��@�(�B�B]�                                    Bxm���  �          A�@�A�\@8Q�A�\)B{�@�@�=q@��B�B]�
                                    Bxm��~  �          A33@�(�@��R@ ��Aw33Bt�R@�(�@���@�  Bz�BX\)                                    Bxm��$  �          A=q@�G�@��@��Ak�
Bp  @�G�@�  @�33B��BS�R                                    Bxm���  �          A�H@�Q�@�Q�@�HAnffBi�@�Q�@���@�33B�RBLp�                                    Bxm��p  
�          A�@�p�@��@!�Aw�Bm
=@�p�@�@��B�BO�                                    Bxm�
  �          A  @���@��@'
=A33BpQ�@���@�ff@��HBBR33                                    Bxm��  �          A��@�33@�p�@#�
A\)Bl(�@�33@�  @�{BBM=q                                    Bxm�'b  �          Az�@�@��@�\Adz�Bb�@�@�\)@�z�B�
BE
=                                    Bxm�6  �          A33@���@�33@ ��A}��Bl�H@���@�ff@��
BBN�                                    Bxm�D�  �          A
=@��R@��R@)��A�ffBu\)@��R@�\)@���B�
BV��                                    Bxm�ST  �          A
=@��@�p�@1�A��RBt�@��@�(�@���B�BS�                                    Bxm�a�  T          A��@�ff@�=q@333A��Bt  @�ff@���@�(�B!(�BS33                                    Bxm�p�  �          A�H@���@��@EA��Bq��@���@�33@��
B'{BM�                                    Bxm�F  �          A�@�{@�
=@K�A��\Bi�@�{@�G�@�=qB(ffBBQ�                                    BxmÍ�  �          A�@���@��@G
=A��BhQ�@���@��@�G�B%�
BA��                                    BxmÜ�  �          A��@���@陚@J=qA���Bm��@���@��@�33B)z�BG�                                    Bxmë8  �          A  @���@��H@333A��RB{p�@���@�Q�@���B"Q�B\G�                                    Bxmù�  �          Az�@k�A=q@   Az{B�Q�@k�@���@�z�B��Bl\)                                    Bxm�Ȅ  �          A�@QG�A{@z�AO�
B�ff@QG�@љ�@��
B
=B~
=                                    Bxm��*  �          A=q@�{@�\@:�HA�{Bg��@�{@�Q�@���B#�BA                                    Bxm���  �          AG�@�@�=q@E�A�33B^(�@�@�ff@�=qB%
=B4�                                    Bxm��v  �          Az�@��@�=q@B�\A�=qB_@��@�
=@���B%33B6�                                    Bxm�  �          Az�@�33@׮@6ffA���BY
=@�33@�
=@��HB(�B0��                                    Bxm��  �          A��@�@�(�@A�A���BUz�@�@���@�ffB!z�B*�                                    Bxm� h  T          A
=@���@ۅ@HQ�A�Q�B\�R@���@�ff@���B%�\B233                                    Bxm�/  �          A33@�{@�@R�\A��HBV
=@�{@��R@��RB'p�B(G�                                    Bxm�=�  �          A�@�Q�@��
@AG�A�p�Bf��@�Q�@�\)@�p�B%��B?33                                    Bxm�LZ  �          A�
@�ff@��H@+�A���Bk�@�ff@��@�
=B33BHQ�                                    Bxm�[   �          A��@��
@��@1�A�
=Bf\)@��
@��R@�G�B�BA��                                    Bxm�i�  �          A�\@��
@�G�@L��A�ffBO�@��
@��@��\B#�\B!��                                    Bxm�xL  �          A=q@���@�33@0��A�\)Bf  @���@���@�ffB �B@                                    BxmĆ�  �          A��@��@�\@,(�A���Bw=q@��@�Q�@��B"�BVp�                                    Bxmĕ�  �          A@}p�@��H@\)Aj=qBy�@}p�@�\)@��RB�B]z�                                    BxmĤ>  �          A
�\@o\)@�\)@��Ak\)B}@o\)@�z�@�(�B�HBa�H                                    BxmĲ�  �          Az�@Fff@���@�A`��B��@Fff@���@�Q�B(�B{G�                                    Bxm���  �          A��@dz�@��@	��Ab�HB�k�@dz�@�(�@�
=B�
Bj�                                    Bxm��0  �          A  @c�
A�?�\A5p�B�p�@c�
@˅@���B�
Bs
=                                    Bxm���  �          A33@n{@��?�(�A  B��@n{@�p�@��BBo��                                    Bxm��|  �          A  @N{A��?���A��B���@N{@�
=@���Bp�B��)                                    Bxm��"  �          A�@��R@�@%�A���Bnff@��R@�@�33B   BK�                                    Bxm�
�  �          A�
@�  @�z�@<(�A��Bcz�@�  @�  @�G�B&(�B:=q                                    Bxm�n  �          A
=@���@��H@��A�
Bh�@���@�@���B�HBF�\                                    Bxm�(  �          A
�R@���@�p�@33Av�RBl{@���@���@��B�
BK\)                                    Bxm�6�  �          Az�@�=q@���@z�At��Bm=q@�=q@�(�@�B�
BL��                                    Bxm�E`  �          A
�H@��@ᙚ@$z�A���Bi  @��@���@���B�BD                                    Bxm�T  �          A
ff@�(�@�{@.{A���Bf�@�(�@�(�@�z�B"�RB@=q                                    Bxm�b�  �          A  @��@޸R@-p�A�=qBb��@��@���@�z�B \)B<{                                    Bxm�qR  �          A
�H@w�@�@G�A���Bu��@w�@��H@\B1�
BM\)                                    Bxm��  �          A(�@��@���@C33A��RBk��@��@���@�\)B,
=BB(�                                    BxmŎ�  �          A��@��@��@9��A�=qBi(�@��@�z�@��
B'\)B@��                                    BxmŝD  T          AQ�@��R@�z�@5A��Bm��@��R@�\)@�33B'p�BG(�                                    Bxmū�  �          A\)@���@�R@2�\A�\)Bs=q@���@��@��HB(Q�BN
=                                    Bxmź�  T          A
=@�G�@�z�@5A���Br  @�G�@�
=@�33B)�BK�R                                    Bxm��6  �          Aff@��@Ӆ@33AaBY�\@��@�(�@��BQ�B7ff                                    Bxm���  �          A@���@љ�?�A8  BT�
@���@�  @���B
=B6Q�                                    Bxm��  �          A=q@��@�ff?���A/�BNQ�@��@�{@�{A��B0                                      Bxm��(  
(          A{@�Q�@ə�?�33A5BH��@�Q�@���@�p�A��B)�                                    Bxm��  �          A�\@��H@ȣ�?�A7
=BFG�@��H@��@�A�=qB&p�                                    Bxm�t  �          A  @��@��@ffAd��BL@��@��@��
B�B(z�                                    Bxm�!  �          A	p�@��@�?�\)AH��BH@��@�G�@�B\)B'
=                                    Bxm�/�  �          A
=q@�G�@�Q�?��A�RB=  @�G�@�(�@�z�A�\)B ��                                    Bxm�>f  �          A  @ʏ\@�{?fff@�ffB,�@ʏ\@��@g
=A�\)B                                    Bxm�M  �          A�R@�{@�33?(�@z�HB$�@�{@��@S�
A�z�B
=                                    Bxm�[�  �          A
=@��@�?��RA   B(\)@��@�(�@z=qAϙ�B                                    Bxm�jX  �          Aff@�(�@�\)?.{@�  B$  @�(�@�{@\(�A��RB�R                                    Bxm�x�  �          A��@ᙚ@�p�>�@A�B=q@ᙚ@�Q�@G�A�
=B{                                    BxmƇ�  �          A  @��H@���>�G�@2�\B�R@��H@���@@��A���B
=                                    BxmƖJ  �          A\)@���@�p�@   AU�B;�\@���@�
=@�ffB
=B��                                    BxmƤ�  T          A\)@��@ҏ\@#�
A�33BQ�R@��@��@�z�Bp�B({                                    BxmƳ�  T          A
ff@�\)@�=q@G�A�(�Bh��@�\)@�  @�  B0�RB:�                                    Bxm��<  �          A
=q@�G�@��@<(�A���Bh�@�G�@��@�(�B+�\B=��                                    Bxm���  �          A	�@�
=@�p�@1�A�\)Bj�@�
=@��@�  B(�BAp�                                    Bxm�߈  �          A	�@�33@��@,��A�ffBg�@�33@���@�B%\)B>p�                                    Bxm��.  �          A  @�@�G�@�Aq��Ba�\@�@��\@�(�B��B=p�                                    Bxm���  �          A
{@��@޸R@'�A��Bg�@��@�33@���B#��B@G�                                    Bxm�z  �          A	�@��\@��
@�RA��RBW�@��\@��@��B��B.�                                    Bxm�   
�          A	@�{@ʏ\@*�HA�z�BJ�@�{@�Q�@�z�B�B�R                                    Bxm�(�  �          A��@���@�=q@�RAq�BRff@���@�@��
B�B+�                                    Bxm�7l  �          A��@��@��
@ffAc�
BS��@��@�G�@�G�B=qB/33                                    Bxm�F  �          A	�@�\)@�p�@�\A\Q�BT�@�\)@��@�Q�B��B1
=                                    Bxm�T�  �          A	�@�Q�@׮?�
=AO
=BUz�@�Q�@��R@�{B�B3                                      Bxm�c^  T          A
{@�\)@�=q?�\)@�ffBH@�\)@��R@��A�z�B.Q�                                    Bxm�r  �          A�H@��R@�@33Aap�B[=q@��R@�33@���B�\B7\)                                    Bxmǀ�  �          A
=@�ff@�=q?���A/33B]��@�ff@�{@��RB�B?p�                                    BxmǏP  �          A�\@�Q�@���?���A�HB[�R@�Q�@�Q�@��B\)B?p�                                    Bxmǝ�  �          A��@���@�{?���A�BQ�@���@�Q�@���A�ffB6�\                                    BxmǬ�  �          A�@�z�@�z�>��
@B8�\@�z�@�@X��A�B%�
                                    BxmǻB  �          A��@�z�@�33>��
@z�B7�@�z�@���@W�A�G�B%=q                                    Bxm���  �          AG�@�  @�ff��p��(�B)�
@�  @�(�@$z�A��Bff                                    Bxm�؎  �          A��@θR@��׾�
=�.�RB+��@θR@��R@$z�A��B �H                                    Bxm��4  �          A@�G�@�{>���@$z�BD�\@�G�@�z�@g
=A��\B1(�                                    Bxm���  �          AQ�@ȣ�@\��\�Tz�B0�@ȣ�@���@!�A�z�B&(�                                    Bxm��  �          A�@Ǯ@Ǯ�����%B4  @Ǯ@�z�@,��A��\B(ff                                    Bxm�&  �          A
ff@Ϯ@���ff�=p�B$��@Ϯ@�p�@��A���B=q                                    Bxm�!�  �          A	�@ָR@�(��&ff���B�@ָR@�Q�@z�A^=qB��                                    Bxm�0r  �          A	G�@�  @���?�R@��
Bi=q@�  @�G�@��A��HBUff                                    Bxm�?  T          Ap�@�
=@Ϯ>B�\?��
BL��@�
=@��@VffA��B;Q�                                    Bxm�M�  �          A�@�@�  ���k�BM��@�@�p�@J=qA�
=B>�                                    Bxm�\d  �          A�@��@�ff>��R@	��BW
=@��@�@dz�A�G�BD��                                    Bxm�k
  �          A��@~{@�\)?aG�@��
Bt�@~{@��@��\A��B_�                                    Bxm�y�  �          A�@�z�@�\?E�@���Bh�H@�z�@���@��A�ffBSG�                                    BxmȈV  �          A@.{@��?�=qA0z�B��H@.{@�(�@���B
=B�Q�                                    BxmȖ�  �          A��@A�@���@�As\)B�k�@A�@��@���B'�Br                                      Bxmȥ�  �          A��@?\)@�@��A��B�Ǯ@?\)@�
=@��RB.  Bp                                    BxmȴH  �          A�@>{@��H?�\)AN�HB�L�@>{@�@�(�B (�By�\                                    Bxm���  �          A�@G
=@�
=?���A.{B�\@G
=@��@��RBBx��                                    Bxm�є  �          A\)@W�@�{?��\@�z�B���@W�@�@�{BBu�                                    Bxm��:  �          A(�@VffA ��?G�@��\B���@Vff@�33@��B ��B|z�                                    Bxm���  �          A(�@`  @��?J=q@�{B��3@`  @�G�@���B \)Bwp�                                    Bxm���  �          AQ�@h��@�ff?z�@r�\B�Ǯ@h��@Ӆ@�ffA�=qBt��                                    Bxm�,  �          A
=q@j�H@�=q>�  ?�33B��R@j�H@��@�33A���Bt��                                    Bxm��  �          A  @Y��@���?�  A�B���@Y��@�G�@�Q�BffBn\)                                    Bxm�)x  �          A�@n�R@�=q?�ff@��B|
=@n�R@��\@��Bz�Be�                                    Bxm�8  �          Az�@�=q@�=q?��
@�z�B`��@�=q@���@�G�A��
BF��                                    Bxm�F�  �          AQ�@��@أ�?fff@�
=B]�
@��@�p�@��A�G�BE=q                                    Bxm�Uj  �          A��@��\@أ�?B�\@��BZ  @��\@�\)@�G�A�BB�                                    Bxm�d  �          A�
@�Q�@�
=>��@QG�BZ@�Q�@��\@p��A�\)BF=q                                    Bxm�r�  �          A��@���@�(�?�=qA33Bq�@���@�Q�@�
=B
Q�BV��                                    BxmɁ\  �          A�@e�@��?�(�A>ffB��@e�@�
=@���B��Bc33                                    Bxmɐ  �          A�@j=q@�\)?�{AO�B|��@j=q@�=q@�\)B�B]�                                    Bxmɞ�  �          A{@n�R@��?�\)AN�RB{Q�@n�R@�33@�Q�B�B\p�                                    BxmɭN  
�          A@q�@�ff?��HAX��ByG�@q�@��@��B��BX��                                    Bxmɻ�  �          A�
@h��@�?��
AG33B|�@h��@�=q@�z�BffB^z�                                    Bxm�ʚ  �          A��@Z=q@��
?�G�AC�B�k�@Z=q@��@�\)B�Bh                                    Bxm��@  �          A��@S33@�ff?�=qAJ{B�G�@S33@���@�=qB�\Bl��                                    Bxm���  �          A{@U�@�ff?�
=AT��B��)@U�@��R@��B ��Bj�                                    Bxm���  �          A(�@b�\@��?�\)AK33B��{@b�\@���@���B�\Be�                                    Bxm�2  �          A�@s33A zῆff��\)B�aH@s33@�R@H��A���B|(�                                    Bxm��  �          A�
@o\)Ap���=q���B�Q�@o\)@���@I��A�Q�B~=q                                    Bxm�"~  �          AQ�@hQ�A�
���Q�B��@hQ�@�(�@n{A¸RBG�                                    Bxm�1$  �          A{@k�A �ÿ���y��B��)@k�@�Q�@c33A���B|\)                                    Bxm�?�  �          A=q@qG�A z���l(�B��@qG�@�R@dz�A�\)By��                                    Bxm�Np  �          A=q@r�\A z��G��6ffB��=@r�\@���@l(�A��Bxff                                    Bxm�]  �          Aff@}p�@�ff�#�
��(�B~�@}p�@�@^{A��Bt�                                    Bxm�k�  �          A\)@��@�{�Y�����RB{�@��@��@R�\A�z�Bs\)                                    Bxm�zb  �          A�@��H@�  ��p����Br��@��H@陚@7�A�G�Bm�                                    Bxmʉ  �          A(�@�G�@�G���Q��33Bt�@�G�@�{@.{A�Q�Bp33                                    Bxmʗ�  �          Aff@��@�  ��  ��Bx=q@��@�{@(��A�=qBtG�                                    BxmʦT  �          A  @~{@�Q�����NffBx�@~{@�R@��Ad  Bx                                      Bxmʴ�  �          A
ff@l��@�
=�
=�}G�B~��@l��@�(�?�\A<Q�B�G�                                    Bxm�à  �          Az�@�(�@������h��BsG�@�(�@�?��AF{Btp�                                    Bxm��F  �          AG�@���@��
��33�,��Bu�R@���@���@p�A�\)Br�H                                    Bxm���  �          A  @��@���ff���Bvz�@��@�R@0��A�33Bq=q                                    Bxm��  �          AG�@z=q@�  �Ǯ�#
=B}  @z=q@�R@'
=A��Byp�                                    Bxm��8  �          A��@l(�@�=q�Ǯ�$  B�k�@l(�@��@)��A�Q�B\)                                    Bxm��  �          A�@g�@����
=�2=qB���@g�@�Q�@ ��A�  B��                                     Bxm��  �          A
�\@S�
@��xQ���B��q@S�
@��@N{A��B�Q�                                    Bxm�**  �          A
=@Q�@����(���B���@Q�@�G�@2�\A��
B�
=                                    Bxm�8�  �          A
=q@c33@�{���2ffB�k�@c33@�R@ ��A���B�{                                    Bxm�Gv  �          A
{@y��@�׿�\)�-��BzQ�@y��@�G�@p�A��\Bw\)                                    Bxm�V  �          A	p�@�G�@�  ��z��2{Bm�@�G�@�=q@33Ax(�Bk
=                                    Bxm�d�  �          A�@c�
@�  ��G��ffB��3@c�
@陚@9��A��B�
=                                    Bxm�sh  �          A\)@2�\A
=��G���(�B�W
@2�\@�G�@UA�z�B�\)                                    Bxm˂  �          A	�@e�@�\)��Q����B�L�@e�@�@<��A�Q�B~��                                    Bxmː�  �          A	G�@g
=@���p���B��R@g
=@�
=@9��A�{B}�
                                    Bxm˟Z  �          A��@Mp�@��H��
=��p�B�ff@Mp�@�=q@AG�A���B���                                    Bxmˮ   �          A	�@>�R@�{����� ��B��f@>�R@�@C�
A��B�L�                                    Bxm˼�  �          A	�@7�@�
=��ff�%��B�L�@7�@��
@1�A���B��                                    Bxm��L  �          A��@6ff@�(���p��:�RB�33@6ff@���@%�A�ffB�{                                    Bxm���  �          A	p�@E@��\��{�-G�B��@E@�G�@*=qA��RB�k�                                    Bxm��  T          A	��@Q�@�녿����p�B�ff@Q�@��
@9��A�Q�B���                                    Bxm��>  �          A	p�@p  @�zΉ����Bff@p  @�@@��A�ffBx�                                    Bxm��  �          A��@J�H@�����C
=B�z�@J�H@��@��A�\)B��                                    Bxm��  �          A	G�@7�@�z��  �;�
B�@7�@��@%A�z�B��H                                    Bxm�#0  �          A(�@^�R@��H��  �"�\B�Ǯ@^�R@��@)��A��B��)                                    Bxm�1�  �          A�@E@�ff��
=�7
=B�=q@E@�ff@#33A�Q�B��                                    Bxm�@|  �          A�@L��@��Ϳ޸R�>{B��=@L��@�ff@{A��B�p�                                    Bxm�O"  �          A�H@Q�@�G������J�HB�\@Q�@��@�A�Q�B�L�                                    Bxm�]�  �          A�\@N�R@��H����4Q�B��@N�R@��H@!�A�=qB��                                    Bxm�ln  �          A�@HQ�@��R�����-B��)@HQ�@�p�@(��A�33B�L�                                    Bxm�{  �          AG�@B�\@񙚿�  �A�B��@B�\@�@�A��B�{                                    Bxm̉�  �          A�
@Q�@�Q��
=�mB�aH@Q�@陚?�(�A\��B���                                    Bxm̘`  �          Aff@U@�(����l��B�Ǯ@U@�?�
=A[�B�                                    Bxm̧  
�          A
=@L(�@����H�@(�B�(�@L(�@�p�@�A��\B�{                                    Bxm̵�  �          A ��@I��@����\)�W\)B��=@I��@��H@Q�Au��B��                                    Bxm��R  �          A z�@_\)@�p��ff�r�\B|�@_\)@�  ?�AP��B}��                                    Bxm���  �          @��
@a�@�{�
�H��  Bx�@a�@��H?��A@  Bz                                    Bxm��  �          @�(�@u�@�������HBhff@u�@Ϯ?�Q�A33Bm                                    Bxm��D  �          @�\@z=q@�G��������Bdff@z=q@���?�{A33Bj\)                                    Bxm���  �          @�ff@{�@����$z����B_��@{�@�G�?W
=@�{Bh
=                                    Bxm��  �          @�\)@S33@������(�Byp�@S33@�=q?���A&=qB}=q                                    Bxm�6  �          @�Q�@W
=@�{�"�\����Bv  @W
=@Ӆ?���A
=B|�                                    Bxm�*�  �          @��@|(�@����ff����BeG�@|(�@�
=?��HA�Bj�\                                    Bxm�9�  �          @��@u�@��H������Bo\)@u�@ə�@z�A��
Bj�                                    Bxm�H(  �          @��
@e�@��H���Bu��@e�@�
=@��A��HBpp�                                    Bxm�V�  �          @�z�@(��@أ��Q���33B�\)@(��@��?\A7
=B��R                                    Bxm�et  T          @�  @'
=@�{�\)��ffB�@'
=@�\?�p�ANffB��=                                    Bxm�t  �          @�\)@#33@�  ��
�w
=B���@#33@�G�?�AeB�(�                                    Bxm͂�  �          @�(�@(�@��Q��{�B�ff@(�@�\)?���Ad��B���                                    Bxm͑f  �          @���@Q�@�{�G��qB��@Q�@�@33At��B��f                                    Bxm͠  �          @�?���@�{���g
=B��q?���@�(�@��A�p�B�z�                                    Bxmͮ�  �          @�Q�?�  @��
��
=�Hz�B�W
?�  @��@(�A��\B��{                                    BxmͽX  �          @�p�?�p�@�z῱��&�\B�L�?�p�@��@-p�A�\)B��                                    Bxm���  �          @�ff?�33@�33��{�A��B�?�33@�33@\)A���B��f                                    Bxm�ڤ  �          @��R?���@����ff�9��B���?���@��
@%�A��HB��                                    Bxm��J  �          @�\)?�z�@���z��F�RB��\?�z�@�(�@p�A��\B�Ǯ                                    Bxm���  �          @�=q?��R@�Q�����~{B�L�?��R@�G�@   Am�B�u�                                    Bxm��  �          @���?���@�(��=q��ffB�k�?���@�\?��HAK
=B�8R                                    Bxm�<  �          @��\?�@��
�
�H��33B�#�?�@��@�\AqB�=q                                    Bxm�#�  
�          @��?�G�@�ff�����f{B�?�G�@�@G�A���B��q                                    Bxm�2�  �          @���?�G�@��H��x��B�?�G�@��H@ffAzffB��q                                    Bxm�A.  �          @�  ?�33@��������B�#�?�33@��
?�Ae�B�p�                                    Bxm�O�  �          @��H?˅@�=q����p�B�p�?˅@�p�?�
=Ad  B�                                    Bxm�^z  �          @�?�p�@��
�����  B���?�p�@��?�\)AY��B��                                    Bxm�m   �          @��
?�(�@�G������{B�\)?�(�@�R?���AUp�B��                                    Bxm�{�  �          @�z�?�Q�@����H����B�
=?�Q�@�\)?�AT��B���                                    BxmΊl  T          AG�?�  @��  ��33B���?�  @���@�As\)B��q                                    BxmΙ  �          @�ff?�  @�\)���mG�B���?�  @�p�@�RA�z�B�p�                                    BxmΧ�  �          @�\)?��@���
=��p�B���?��@�G�?�A_
=B�8R                                    Bxmζ^  �          A (�?�\)@�\)�
=���B���?�\)@�33?��HAbffB���                                    Bxm��  �          A ��?�=q@����Q����B�W
?�=q@�z�?�(�Ab=qB��                                    Bxm�Ӫ  �          A Q�?�ff@���z���z�B��R?�ff@�@G�Ai�B���                                    Bxm��P  �          A (�?���@�����  B�(�?���@��
@   Ag�B�k�                                    Bxm���  �          @��?�ff@�{�&ff��{B�#�?�ff@�{?޸RAIG�B�                                    Bxm���  �          A ��?:�H@�
=�G��j{B�#�?:�H@��
@Q�A�=qB�                                      Bxm�B  �          @�z�?��@�\)�
=��\)B��)?��@��H?�p�Ah��B�                                      Bxm��  �          @��H?h��@�p���
��Q�B�?h��@��H?�(�AXQ�B�aH                                    Bxm�+�  �          @�>�ff@߮�����ffB���>�ff@�\)?��AK�B��
                                    Bxm�:4  �          @�R>�G�@߮�#�
����B��)>�G�@���?��A?\)B�#�                                    Bxm�H�  �          @�(�?�\@�p��!G�����B���?�\@�ff?�ffAAG�B���                                    Bxm�W�  �          @�?�ff@�ff��
���B�?�ff@��
?޸RAY�B�(�                                    Bxm�f&  �          @�{?}p�@���$z���p�B���?}p�@�
=?�  A:ffB�L�                                    Bxm�t�  �          @�{?fff@�\)������RB�L�?fff@�{?�
=AP��B��R                                    Bxmσr  �          @���?��\@�\�����B���?��\@�R?�=qAb=qB��f                                    Bxmϒ  �          @�\)?z�H@����p���(�B�.?z�H@��
?�\)AiB�aH                                    BxmϠ�  
�          @ᙚ?��@��
�.�R��(�B�?��@�33?�=qAG�B�W
                                    Bxmϯd  �          @�(�?�=q@�{�,(���{B���?�=q@�p�?��\A��B��                                    BxmϾ
  �          @��
?���@�p��,(���ffB�aH?���@���?�G�A�
B��                                    Bxm�̰  �          @�{?У�@����/\)��=qB�G�?У�@��?u@�\)B�z�                                    Bxm��V  �          @޸R?�ff@�33�<(��ȏ\B�k�?�ff@�\)?B�\@�G�B�                                      Bxm���  �          @�\)?�(�@Ϯ�333���\B��?�(�@߮?��A\)B�B�                                    Bxm���  �          @��?��@�  �333��\)B��q?��@�  ?���A\)B���                                    Bxm�H  �          @�p�?Ǯ@���� ����{B�.?Ǯ@ۅ?�{A/33B�p�                                    Bxm��  �          @�=q?��
@Ϯ�
=��{B�u�?��
@�  ?�(�A?�B�p�                                    Bxm�$�  �          @��H?�  @������Q�B�ff?�  @��?�  AC�B�.                                    Bxm�3:  �          @�ff?�33@θR�	�����B���?�33@Ӆ?��AY�B�#�                                    Bxm�A�  �          @ᙚ?�  @���������B�G�?�  @�Q�?\AG
=B�                                    Bxm�P�  �          @��?�@˅�=q��p�B�=q?�@�p�?�{A6=qB�(�                                    Bxm�_,  T          @�G�?�33@���&ff��{B��f?�33@��H?���A�B�B�                                    Bxm�m�  �          @޸R?���@�33�(����=qB�k�?���@���?�z�A�B���                                    Bxm�|x  �          @�33?�z�@��0  ����B�� ?�z�@���?�{Az�B��f                                    BxmЋ  �          @��?�G�@����Q��֣�B�#�?�G�@��?(��@�B�                                      BxmЙ�  �          @��?u@θR�K��ϙ�B��?u@��?E�@�33B��q                                    BxmШj  �          @�  ?�\)@�
=�E�ɮB�k�?�\)@�?Y��@�  B�(�                                    Bxmз  �          @�
=?k�@���W���z�B�u�?k�@�z�?
=q@���B�W
                                    Bxm�Ŷ  �          @�R?�
=@����Vff���B���?�
=@�33?
=q@�Q�B�\                                    Bxm��\  �          @��?B�\@�  �Z=q��33B�\?B�\@�33>��@r�\B��R                                    Bxm��  �          @�p�?333@�ff�a��뙚B�#�?333@�(�>�33@1�B���                                    Bxm��  �          @�(�?8Q�@�z��aG����HB���?8Q�@�=q>��
@'
=B�W
                                    Bxm� N  �          @�Q�?
=q@�\)�hQ����B��?
=q@߮>\)?�z�B�(�                                    Bxm��  T          @�\)=#�
@�{��33�
��B�#�=#�
@�R��\)��RB�L�                                    Bxm��  T          @��H��@�������B����@陚����j�HB��f                                    Bxm�,@  �          @��;�z�@��R��33�Q�B�� ��z�@��
����k�B�u�                                    Bxm�:�  �          @��;aG�@�33�����  B�aH�aG�@��0�����\B�z�                                    Bxm�I�  �          @�녽���@�ff���R�G�B��H����@��þ�p��8Q�B��=                                    Bxm�X2  �          @�Q�B�\@���\)�{B��
�B�\@�ff�=p����\B�
=                                    Bxm�f�  �          @�  ��G�@ȣ��g���(�B�
=��G�@�\)>��R@�B�                                    Bxm�u~  �          @�p�=�@ȣ��]p�����B�=�@���>�ff@eB�
=                                    Bxmф$  �          @�z�>�=q@�  �W���RB�{>�=q@��H?�\@��B���                                    Bxmђ�  �          @��
>aG�@��H�Mp��֣�B���>aG�@�=q?333@�(�B�aH                                    Bxmѡp  �          @���>�Q�@�z��P������B��q>�Q�@�33?O\)@�=qB�\)                                    BxmѰ  �          @�{>�Q�@ڏ\�`���مB��>�Q�@�z�?333@�Q�B���                                    BxmѾ�  �          @��>��@ڏ\�g����B�33>��@��R?�R@��B���                                    Bxm��b  �          @��\>�G�@޸R�c33���
B���>�G�@���?=p�@��RB��{                                    Bxm��  T          @���>�\)@����\)����B�8R>�\)@�Q�>W
=?��
B��f                                    Bxm��  �          @�\)���
@�\)������HB�p����
@�\)<#�
=#�
B�8R                                    Bxm��T  �          A{���@Ӆ����z�B�� ���A녾�G��G�B���                                    Bxm��  �          A��Tz�@��H��p���
B�  �Tz�A{�s33����B��q                                    Bxm��  �          A   ��=q@Å��  �z�B�
=��=q@��H�n{��
=Bî                                    Bxm�%F  �          @�?�  @�G��%���(�B�#�?�  @���?��A,Q�B�p�                                    Bxm�3�  �          @�ff@ff@��
��\)�p��B��q@ff@��H?��RA�Q�B��\                                    Bxm�B�  �          @�@G
=@ʏ\�����RB=q@G
=@�ff@(�A�  Byz�                                    Bxm�Q8  �          @�\)@>�R@θR��  ��
B�� @>�R@��
@�HA�z�B��                                    Bxm�_�  �          @���@-p�@��ÿ�p��[�
B��@-p�@�{@�
A�G�B�Q�                                    Bxm�n�  �          @�@�R@�(����R�z�RB��@�R@��?��An�RB�{                                    Bxm�}*  �          @�
=@
�H@ڏ\�ff���RB���@
�H@�z�?�33AlQ�B�{                                    Bxmҋ�  �          @���?8Q�@��H�Dz���z�B�\)?8Q�@�(�?�p�A��B�.                                    BxmҚv  
�          @��
?L��@���Dz���z�B��R?L��@�\)?���A��B��R                                    Bxmҩ  �          @��?�@ٙ��_\)�؏\B�.?�@�?8Q�@��B��                                    Bxmҷ�  �          A녿(��@Ӆ�����
B�k��(��A
=�
=q�qG�B�=q                                    Bxm��h  �          Az῀  @љ���(��(�B�LͿ�  A�R��  ��  B��                                    Bxm��  �          A(����
@�{�����HB�uÿ��
Ap������B�p�                                    Bxm��  �          A	p���z�@��
����ffB��ÿ�z�A\)�u����Bè�                                    Bxm��Z  �          A	��@�=q����B�33��A\)�����
=Býq                                    Bxm�   �          A��
=@�G���(�� (�B�33��
=A
=������
B�\)                                    Bxm��  �          A�R���\@����=q��B�\)���\AQ쿅����B�k�                                    Bxm�L  �          A�\��p�@�=q��p�� p�B��῝p�A�
����
=B��                                    Bxm�,�  �          Aff��p�@�
=��Q��$
=B�LͿ�p�A\)��ff�{B���                                    Bxm�;�  �          A=q����@�{���!p�B�z����A=q���R�(�B��)                                    Bxm�J>  �          A�H��@������\�%��B�녿�A
=�����BǞ�                                    Bxm�X�  
�          A
=��z�@�=q���)G�B����z�A�R��G��%p�BǊ=                                    Bxm�g�  �          A\)��z�@�ff��33�.��BΞ���z�A�R��p��<z�BǏ\                                    Bxm�v0  �          A�׿�  @�=q��p��-G�B���  @��R����7\)B�aH                                    Bxmӄ�  �          A(��\@�
=�����1{BѸR�\@����
�G33Bɮ                                    Bxmӓ|  �          A(��Ǯ@�{�����;  B�{�Ǯ@��\����pQ�Bʙ�                                    BxmӢ"  �          A�ÿ��@�{��\)�'�RB�녿��A�׿��R� ��B�                                      BxmӰ�  �          A�Ϳ�ff@�{����$33B�\)��ffA�
��33�ffB̙�                                    Bxmӿn  �          A�׿�z�@�p���p��&Q�B�{��z�A  �������BʸR                                    Bxm��  �          A  ��33@Å����)�
Bͳ3��33A�
����'\)B�8R                                    Bxm�ܺ  �          A�
�У�@�p����H�$�HBя\�У�A\)�����RB�p�                                    Bxm��`  �          A�H����@�Q����\��B��
����Aff��\)��z�B͊=                                    Bxm��  �          A
=��\@������{B�B���\A�H��=q���HB�\)                                    Bxm��  �          A�\��ff@�p���ff�*�B�Ǯ��ffA �ÿ�{�0��B�(�                                    Bxm�R  �          A�R��@��H����%�Bҽq��A{��z��=qB�G�                                    Bxm�%�  T          A33��@�{���\�(�B�G���AG���z����B�(�                                    Bxm�4�  �          A
=����@��
��33��B�p�����A  ��=q��33B�33                                    Bxm�CD  �          A
=��(�@�
=��
=��HB͏\��(�A(��k���Q�B�#�                                    Bxm�Q�  �          Azῠ  @������H��Bɨ���  A녿}p�����B���                                    Bxm�`�  �          A�
�h��@�\)����Q�B�\�h��A녿�����  B�u�                                    Bxm�o6  �          A=q�aG�@�\)��  ��B�z�aG�Az�p����ffB�#�                                    Bxm�}�  �          Aff�h��@������BÅ�h��AG��#�
���B�u�                                    BxmԌ�  	�          A��fff@�(���=q���B�uÿfffA��5���\B�\)                                    Bxmԛ(  �          A��@�
=��ff�+�B��Ϳ�Ap��˅�/\)B��                                    Bxmԩ�  �          A�R��{@�\)��{�3�RB�p���{A Q���S�BʸR                                    BxmԸt  �          A\)����@��R�����5��B�=q����A ���   �[�B�k�                                    Bxm��  T          A�
����@�\)��G��#  B��)����A������\)B�
=                                    Bxm���  
�          A�
�˅@�  �����,G�Bѽq�˅A�H���4��B�
=                                    Bxm��f  
�          A���@�Q�����3�HB�\)��AG������UG�B�aH                                    Bxm��  �          A  ��=q@�ff��33�7�\B�\��=qAG����b�RB�(�                                    Bxm��  T          AQ��  @����  �?�\B֔{��  A��{���RB���                                    Bxm�X  �          A�H���@��
��
=�B��B�W
���A���(�����B�33                                    Bxm��  T          A�R��z�@��H��  �D��B�\��z�Ap��,(���(�Bʀ                                     Bxm�-�  �          A��
=@�G���ff�D��B���
=Az��*�H��z�B���                                    Bxm�<J  �          A��У�@�G���G��B=qBԸR�У�A
=�"�\��\)B�z�                                    Bxm�J�  "          A
ff�˅@��H���
�G��B�.�˅A ���.{���\B�Q�                                    Bxm�Y�  �          Az�˅@�=q��
=�Ez�B�=q�˅@�
=�&ff��{Bʏ\                                    Bxm�h<  T          Az῵@�{�Ϯ�Nz�B����@���4z���\)B�
=                                    Bxm�v�  �          A33���@�Q������^�RB��)���@�ff�S33��p�B��
                                    BxmՅ�  �          AG��
=@������
�jffB�LͿ
=@�{�dz���z�B�u�                                    BxmՔ.  
�          A z��  @���ƸR�J��B�B���  @��(Q����HB���                                    Bxmբ�  �          A녿�\)@��R��
=�Sp�B�W
��\)@�ff�;����B�Ǯ                                    Bxmձz  �          @񙚿��@�ff���O��B�p����@�{�'
=��ffB���                                    Bxm��   T          @��ÿ�33@����
�I�
B����33@�  ����B�p�                                    Bxm���  
�          @�33���R@������M\)B�#׿��R@�  �(�����B�                                    Bxm��l  
�          @�z῜(�@z�H��G��UBր ��(�@����$z����B��                                    Bxm��  
Z          @�녿��@u��Q��X{B�����@�{�&ff��33B��)                                    Bxm���  
�          @ٙ����@s�
�����Y�\B�8R���@��'����BǏ\                                    Bxm�	^  �          @ٙ���33@o\)����[�\B�.��33@�(��,(����\B�8R                                    Bxm�  
Z          @�33����@l�������Z(�B�33����@\�,(���z�B�{                                    Bxm�&�  
�          @��>�R@��H��
=�#G�B��f�>�R@�����
�C�B���                                    Bxm�5P  �          @�\)�@  @�p����
�)Q�B�u��@  @��H�޸R�_\)B�q                                    Bxm�C�  �          @�33�e@����  � �CǮ�e@�{�ٙ��U�B�G�                                    Bxm�R�  �          @�=q�g�@r�\���\�0\)C���g�@�p������Q�B��f                                    Bxm�aB  T          @���_\)@qG���{�5(�C�
�_\)@��R����G�B�                                    Bxm�o�  "          @陚�_\)@e�����:CL��_\)@���$z����HB�\                                    Bxm�~�  �          @���qG�@L(���Q��@��C���qG�@�z��<����{B��=                                    Bxm֍4  �          @��n�R@I������Fz�Cٚ�n�R@�\)�J�H����B�#�                                    Bxm֛�  
�          @�z��{@C33��G��xB�׿�{@�=q�x����Q�BѮ                                    