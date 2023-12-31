CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230715000000_e20230715235959_p20230716021656_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-07-16T02:16:56.302Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-07-15T00:00:00.000Z   time_coverage_end         2023-07-15T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�m��  r          AZ�H@�Q��O����;�(�C��@�Q��O����
=qC��                                    Bx�m�f  T          AZ{@�33�K����;�
=C�o\@�33�K�����\C�o\                                    Bx�m�  �          AY@��H�K
=�=p��HQ�C�p�@��H�J�H�B�\�Mp�C�p�                                    Bx�m��  T          AXz�@��G33��{����C���@��G33��\)���RC��                                    Bx�m�X  T          AX  @��\�EG��
=��
C�0�@��\�EG������C�1�                                    Bx�m��  T          AV�H@���G33�333�@��C�<)@���G
=�3�
�A��C�=q                                    Bx�m�  
Z          AW�@o\)�L  ��
��C�.@o\)�L  �z����C�.                                    Bx�m�J            AYG�@|���H���^�R�l��C���@|���H���^�R�l��C���                                    Bx�n�  T          AY@��6{��33��Q�C��{@��6{��33��Q�C��{                                    Bx�n�  
�          AX(�@����1p�����z�C��@����1p�����Q�C��                                    Bx�n <  T          AW�
@����=��{���C���@����=������C���                                    Bx�n.�  �          AVff@����A��K��[�
C���@����A��J�H�[33C���                                    Bx�n=�  �          AW\)@�{�Dz��2�\�?�C�q�@�{�Dz��1G��>�RC�p�                                    Bx�nL.  �          Ai�@θR�J�H@Tz�AQ�C�~�@θR�J�R@U�AS
=C��                                     Bx�nZ�  "          Aj{@����L��@ffAQ�C���@����L��@�A��C���                                    Bx�niz  �          Aj{@Ϯ�Pz�?�p�@�=qC�:�@Ϯ�Pz�?�G�@��C�<)                                    Bx�nx   �          Ai�@�z��T��?��R@��C��@�z��T��?��\@�
=C���                                    Bx�n��  "          Ai��@����Pz�?�z�@�G�C�޸@����Pz�?���@��C�޸                                    Bx�n�l  �          Ah��@�{�M@�A
{C�N@�{�M��@{AQ�C�O\                                    Bx�n�  �          Ah(�@Ϯ�L  @=qA��C�|)@Ϯ�K�
@(�A\)C�}q                                    Bx�n��  �          Ag�@���G�@G�AG33C��@���G\)@J=qAI�C�\                                    Bx�n�^  
�          AdQ�@�G��@z�@c�
Ag
=C���@�G��@Q�@fffAi�C���                                    Bx�n�  �          A`��@��C�@K�AQ�C�k�@��C\)@N�RATz�C�o\                                    Bx�nު  
�          AX��@�z��C33?�\)@�z�C�e@�z��C33?�
=@�C�ff                                    Bx�n�P  
�          ATz�@���G�?�\@p�C��)@���G�?�@��C��)                                    Bx�n��  �          AW�@����K
=?�R@*=qC��\@����J�H?.{@:�HC��\                                    Bx�o
�  
�          AX��@fff�N�\?�G�@�\)C��@fff�Nff?�=q@�Q�C��\                                    Bx�oB  T          AT��@fff�L  �(��*=qC��H@fff�L(�����
=C��                                     Bx�o'�  �          AN=q@s33�C
=�����z�C��=@s33�C33��(���=qC���                                    Bx�o6�  �          AS�
@�  �B{?��\@���C�5�@�  �A�?���@��
C�7
                                    Bx�oE4  
�          ATz�@�=q�<��@
=qAffC���@�=q�<z�@�RA�C���                                    Bx�oS�  �          AU��@�Q��>�H?�{@�  C�c�@�Q��>�R?�Q�@�33C�ff                                    Bx�ob�  
�          AU�@���<��@$z�A0��C��{@���<��@)��A6�RC���                                    Bx�oq&  
�          AS�@���<��@�
A ��C�u�@���<��@��A'33C�y�                                    Bx�o�  �          AP(�@��R�@��?�Q�@��C�'�@��R�@��?��@�C�*=                                    Bx�o�r  �          AN�H@S�
�H  >��?&ffC�l�@S�
�G�
>�  ?���C�l�                                    Bx�o�  �          AO�@%�J�R?:�H@Mp�C�Ǯ@%�J�R?Tz�@k�C���                                    Bx�o��            AO�@%��K
=?fff@\)C��q@%��J�H?�G�@�\)C���                                    Bx�o�d  �          AP��@#33�L(�?c�
@z=qC���@#33�L(�?�G�@�p�C���                                    Bx�o�
  �          AP��@��LQ�?�p�@�z�C�aH@��L(�?���@�p�C�b�                                    Bx�oװ  T          AR=q@3�
�LQ�?��@���C�7
@3�
�L  ?�@�ffC�8R                                    Bx�o�V  
�          ATQ�@=p��Mp�?���@�G�C�}q@=p��MG�?ٙ�@�33C��                                     Bx�o��  
�          ARff@��N�\?L��@^�RC��@��Nff?n{@�=qC�)                                    Bx�p�  �          AS�
@A��M�?Tz�@g
=C���@A��L��?xQ�@�
=C���                                    Bx�pH  T          AV�\@q��L  ?��@��C�@ @q��K�?��
@�(�C�C�                                    Bx�p �  
�          A\(�@���I�@Q�A]p�C���@���Hz�@[�Ag�C���                                    Bx�p/�  �          A[�@���G\)@`  Al  C�(�@���F�\@h��Av=qC�0�                                    Bx�p>:  �          A[33@��R�E�@r�\A�Q�C�t{@��R�DQ�@|(�A���C�}q                                    Bx�pL�  �          AZ{@�=q�B�R@��HA���C�G�@�=q�A�@��A�
=C�Q�                                    Bx�p[�  "          AYp�@��=�@xQ�A���C�U�@��<Q�@���A�=qC�aH                                    Bx�pj,  �          AYp�@�
=�<  @o\)A\)C��)@�
=�;33@x��A�33C��                                    Bx�px�  �          AZ{@�(��?
=@W�Aep�C�� @�(��>=q@b�\Ap��C���                                    Bx�p�x  
�          AX(�@����F=q@=qA$��C���@����E��@%A0��C���                                    Bx�p�  
�          AX��@��H�F�H@5AA�C�!H@��H�F{@AG�ANffC�(�                                    Bx�p��  �          AX��@�  �G�@1�A=G�C���@�  �F�H@=p�AJ{C��{                                    Bx�p�j  T          AX��@�G��C�@8��AE�C�/\@�G��C
=@E�AR{C�9�                                    Bx�p�  
�          AW�
@���C33@ ��A+�
C�` @���Bff@,��A9�C�h�                                    Bx�pж  �          AX  @����B=q@
=A!�C��)@����A��@#33A.�\C�                                    Bx�p�\  "          AW�
@��\�A�@"�\A-�C��q@��\�AG�@/\)A;�C��                                    Bx�p�  T          AU�@�\)�AG�@$z�A1�C�0�@�\)�@��@1G�A@(�C�9�                                    Bx�p��  �          AT(�@��H�@��@0  A?33C��{@��H�?�
@<��AM�C���                                    Bx�qN  
�          AS�
@��R�@z�@=qA'�C�33@��R�?�@'�A6�\C�<)                                    Bx�q�  
�          AS\)@�ff�>�R@(�A��C��=@�ff�>{@=qA((�C��3                                    Bx�q(�  
�          AQp�@��H�<Q�@z�Ap�C�1�@��H�;�@�A ��C�<)                                    Bx�q7@  "          AR�\@��
�=p�@�
A�
C�4{@��
�<��@�A�C�=q                                    Bx�qE�  �          AT(�@���=G�@33A�RC��\@���<��@�A�RC���                                    Bx�qT�  T          AT��@�
=�>{?��H@��
C���@�
=�=p�?�Q�A=qC���                                    Bx�qc2  
�          AT��@�(��;33?�z�@��HC��{@�(��:�R?�33@��
C��)                                    Bx�qq�  �          AUG�@��7�?��@��C�)@��733?���@�C�#�                                    Bx�q�~  
Z          AU�@ۅ�6�\?��@��C�� @ۅ�6=q?���@�{C���                                    Bx�q�$  T          AT��@�G��8(�?��
@�  C�˅@�G��7�?\@�=qC��{                                    Bx�q��  T          AT��@Ϯ�8z�?�
=@�C���@Ϯ�7�
?�Q�@��C��)                                    Bx�q�p  T          AT��@�33�9��?���@ȣ�C�Z�@�33�9�?��H@�z�C�c�                                    Bx�q�  �          AUG�@�z��<  ?��@�z�C�˅@�z��;�?�=q@�G�C��{                                    Bx�qɼ  �          AT(�@���<��?˅@�(�C�8R@���<  ?�{A ��C�AH                                    Bx�q�b  T          AS
=@�(��:�R?�\@�ffC�aH@�(��9�@�\A�\C�l�                                    Bx�q�  
�          APQ�@��:ff@ffA(�C�~�@��9p�@Q�A((�C���                                    Bx�q��  T          AK�
@��R�7\)?�\A Q�C�9�@��R�6�R@33A��C�E                                    Bx�rT  
(          AL  @�G��5�?�
=@�{C�  @�G��5G�?��H@�\)C��                                    Bx�r�  �          AJ�R@��
�2{?c�
@�=qC���@��
�1?�
=@�33C��\                                    Bx�r!�  
�          AL��@����:�R@"�\A7
=C�޸@����9��@5AL��C��                                    Bx�r0F  
�          AL��@�33�733@mp�A��RC�h�@�33�5��@�Q�A��C�~�                                    Bx�r>�  T          AK�
@�p��6{@z�HA�p�C�\@�p��4Q�@�
=A���C�&f                                    Bx�rM�  �          AM�@n�R�9@~{A�(�C��f@n�R�7�
@���A�  C��)                                    Bx�r\8  "          ALQ�@p���;�
@QG�Am�C�� @p���:ff@fffA���C���                                    Bx�rj�  "          ALz�@u��:=q@`��A�  C�)@u��8��@uA�=qC�/\                                    Bx�ry�  �          AJ=q@����6�R@hQ�A��C��{@����4��@}p�A�  C��=                                    Bx�r�*  
�          AI��@8���8Q�@�{A��C��@8���6=q@���A�ffC�                                      Bx�r��  �          AJff@Fff�9G�@�Q�A�  C�|)@Fff�733@�33A�
=C��\                                    Bx�r�v  
�          AJ�R@g��9G�@c33A�z�C��\@g��7�@y��A���C���                                    Bx�r�  
�          AK33@��\�4(�@z�A(��C��=@��\�2�H@*=qAB=qC��q                                    Bx�r��  "          AJ�R@�G��7\)?��HA=qC�� @�G��6=q@�
A((�C��\                                    Bx�r�h  �          AI�@��H�5�?Ǯ@�C��H@��H�4Q�?�A�C��\                                    Bx�r�  �          AK�@��5?��Ap�C�Ǯ@��4��@	��A�C��R                                    Bx�r�  �          AJ=q@�p��4��@=qA/�
C�N@�p��333@1�AJ�HC�c�                                    Bx�r�Z  T          AI@���8  @ffA,  C��@���6�R@.�RAH  C�                                      Bx�s   T          AI�@�
=�8��@=qA0z�C��3@�
=�7\)@333AM�C��f                                    Bx�s�  �          AI@x���;\)@#�
A;�C�.@x���9@=p�AY�C�@                                     Bx�s)L  z          AIp�@�ff�7�@B�\A_�C��@�ff�5@[�A}�C�#�                                    Bx�s7�  `          AD��@���2{@J=qAn�HC��@���0(�@b�\A�ffC�!H                                    Bx�sF�  
�          AJ�H@����2�H@���A���C���@����0z�@�{A�  C��H                                    Bx�sU>  �          AC33@����-�@g
=A�  C��R@����*�H@�  A�33C��R                                    Bx�sc�  T          AA@���,z�@`��A�
=C���@���*ff@y��A�z�C�ٚ                                    Bx�sr�  �          A?\)@�
=�,Q�@#�
AF=qC�E@�
=�*�R@=p�AeG�C�]q                                    Bx�s�0  "          A@  @�{�*{@VffA��
C�XR@�{�'�
@p  A���C�y�                                    Bx�s��  
�          A@��@�p��,z�@?\)Af�\C�#�@�p��*�\@Y��A�G�C�AH                                    Bx�s�|  �          A?33@��+33@8Q�A_�
C�>�@��)G�@R�\A�{C�Z�                                    Bx�s�"  
�          ADQ�@��H�/�@S�
Az�RC�Ǯ@��H�-p�@n�RA��C��                                    Bx�s��  "          AA�@q��"�R@�  A��C�/\@q��
=@���A�  C�e                                    Bx�s�n  T          A>�R@j�H���@���A�
=C�w
@j�H���@���B
=C��R                                    Bx�s�  T          AB=q@���%G�@��A�ffC�S3@���"=q@���A�p�C���                                    Bx�s�  T          AH(�@�
=�333@@��A^�RC�l�@�
=�1�@^{A���C���                                    Bx�s�`  
Z          AF�\@��R�(z�@w�A��
C�*=@��R�%@���A���C�Y�                                    Bx�t  
Z          AG33@�Q��$��@�{A�  C�3@�Q��!�@��
A��HC�J=                                    Bx�t�  
�          AG\)@�33�%G�@r�\A��
C���@�33�"�\@�\)A��RC��R                                    Bx�t"R  �          AH  @�z��$��@w�A��\C��H@�z��"{@��A���C�R                                    Bx�t0�  �          AH(�@����%�@aG�A��RC�c�@����"ff@}p�A�C��
                                    Bx�t?�  T          AI�@��
�)��@/\)AIC�H@��
�'�@L��Al��C�(�                                    Bx�tND  
�          ALQ�@�{�,Q�@�A(��C�s3@�{�*ff@3�
AK�C���                                    Bx�t\�  T          AL��@�  �1G�?У�@���C��R@�  �/�
@�Az�C�Ф                                    Bx�tk�  T          AL��@���333?��\@�C�4{@���2{?��
@�\)C�G�                                    Bx�tz6  �          ALQ�@�Q��333?��@��
C��@�Q��2{?�z�@�ffC�0�                                    Bx�t��  
�          AL��@����3\)?�(�@��RC�q@����2ff?޸R@��C�0�                                    Bx�t��  T          AK33@���5����Q��\)C�B�@���5��>8Q�?L��C�AH                                    Bx�t�(  T          AJff@�(��6ff�h�����
C���@�(��6�H��Q���C���                                    Bx�t��  "          AI��@�(��4�Ϳ����ə�C��R@�(��5�Tz��r�\C���                                    Bx�t�t  T          AH  @���3\)������C���@���4Q쿃�
���RC��)                                    Bx�t�  �          AF=q@���2=q���H��{C�O\@���3\)��z����
C�=q                                    Bx�t��  T          AC�
@�p��0�Ϳs33��ffC��=@�p��1G����Ϳ�{C���                                    Bx�t�f  �          AAp�@��/������{C�{@��0Q�
=q�"�\C�
=                                    Bx�t�  �          A@z�@����-G��W
=��  C��
@����-��z΅{C��\                                    Bx�u�  �          A@��@����/���ff��\)C��@����0(�����C�H                                    Bx�uX  
�          A@Q�@����/�
�z�H��{C���@����0Q��녿���C���                                    Bx�u)�  �          A@Q�@��
�,Q쿹�����C��{@��
�-G��aG���C���                                    Bx�u8�  �          A@Q�@���*ff������C��@���+���(����C��                                    Bx�uGJ  T          A@(�@��R�)G���R�+�C�)@��R�*�H��z����RC�                                      Bx�uU�  �          A?�@����+33��ff���C�ٚ@����,Q�u��33C�Ǯ                                    Bx�ud�  
�          A?\)@�33�*=q��\)�Q�C��@�33�+������C��
                                    Bx�us<  �          A9p�@�(��)p��Q����C��@�(��)녾k����C��f                                    Bx�u��  
�          A9G�@����(�׽�\)��{C�5�@����(Q�?�@&ffC�8R                                    Bx�u��  T          A9��@��#�>�?#�
C��q@��#33?5@`��C�                                    Bx�u�.  
�          A;\)@����)p�?J=q@w�C�4{@����(Q�?��@ۅC�E                                    Bx�u��  "          A:�H@�p��+
=?0��@Y��C���@�p��*{?��@�{C��)                                    Bx�u�z  T          A:�R@����(��?�G�@�\)C�#�@����'�?��Az�C�<)                                    Bx�u�   	�          A;�@�p��)�?�z�AffC���@�p��'
=@!�AG�
C�\                                    Bx�u��  
�          A9��@�G��&�\?��A�C�Y�@�G��$z�@=qA?�C�z�                                    Bx�u�l  
�          A8��@�z��%@�A!�C�\@�z��#�@)��ATQ�C�4{                                    Bx�u�  
�          A8  @�
=�(  ?�(�@�(�C��=@�
=�&�\?�{AG�C��H                                    Bx�v�  �          A:�R@�
=�-p�?��@.{C��H@�
=�,��?�(�@�  C��                                    Bx�v^  T          A8  @����$��?��@��RC�xR@����#
=@
�HA.�HC��R                                    Bx�v#  T          A:�H@����\)@"�\AI�C��@����z�@J=qA{33C���                                    Bx�v1�  "          A:�\@�33� ��@!G�AH(�C�f@�33�@I��A{33C�>�                                    Bx�v@P  
Z          A;
=@�p�� (�@�\A ��C�@�p��@*�HAS\)C���                                    Bx�vN�  "          A<z�@�G��#�?��A33C�=q@�G��!G�@"�\AG
=C�h�                                    Bx�v]�  "          A<��@����%�?��A33C���@����#�@#�
AH(�C���                                    Bx�vlB  
�          A;
=@���&�\?�(�@�C��R@���$��@	��A)�C��                                    Bx�vz�  T          A6=q@�(��%?E�@w�C��@�(��$��?���@�\C�)                                    Bx�v��  T          A7�@����*=q>�=q?�{C��@����)��?z�H@�p�C���                                    Bx�v�4  
�          A;33@��R�.�\�\)�.{C��@��R�.=q?��@:�HC���                                    Bx�v��  
�          A:{@|(��.�H�W
=���\C��@|(��.�R?
=q@(��C��=                                    Bx�v��  T          A9@�(��-p�=�?z�C�l�@�(��,��?\(�@�Q�C�t{                                    Bx�v�&  	�          A:=q@���+�?333@\(�C�]q@���*=q?�Q�@��HC�o\                                    Bx�v��  �          A9��@�(��+\)?0��@[�C��@�(��*=q?�Q�@�(�C�0�                                    Bx�v�r  �          A9�@�  �+�?�\@!G�C��@�  �*�R?�G�@ȣ�C��)                                    Bx�v�  
�          A9G�@�  �,(�>\?�{C��f@�  �+\)?��@���C��3                                    Bx�v��  �          A9��@��
�)?B�\@q�C���@��
�(Q�?\@�G�C��R                                    Bx�wd  �          A9@����+�
>\?�{C�!H@����+
=?�33@�{C�.                                    Bx�w
  
�          A8��@�G��-�<#�
=#�
C�:�@�G��,��?L��@}p�C�AH                                    Bx�w*�  
Z          A<Q�@`���3\)�=p��eC��{@`���3�=�Q�>�(�C���                                    Bx�w9V  
�          A>{@_\)�5G��J=q�tz�C���@_\)�5=L��>uC���                                    Bx�wG�  "          A=@�G��0�׾�{�У�C��q@�G��0��>��H@
=C���                                    Bx�wV�  
�          A<��@��\�)��?}p�@��C���@��\�'�
?��
A33C���                                    Bx�weH  "          A=�@�G��*=q?n{@�  C���@�G��(��?�p�A�HC�Ǯ                                    Bx�ws�  �          A=��@�z��0  >\)?0��C�޸@�z��/\)?}p�@���C���                                    Bx�w��  �          A>�\@��H�1��<�=�C��@��H�0��?aG�@�  C��
                                    Bx�w�:  
�          A<��@�ff�0Q�<�>\)C�o\@�ff�/�
?c�
@��C�xR                                    Bx�w��  "          A=��@�33�0Q�=���?   C���@�33�/�?xQ�@�C���                                    Bx�w��  T          A<��@�p��.�H<#�
=uC�H@�p��.ff?aG�@���C��                                    Bx�w�,  
�          A=�@��H�/33���
��Q�C�aH@��H�.�\?\(�@�(�C�h�                                    Bx�w��  �          A>�R@�p��/�>�z�?�z�C��@�p��.�\?�@���C��3                                    Bx�w�x  T          A?33@���0Q�>W
=?�  C�Z�@���/�?���@���C�g�                                    Bx�w�  �          A>�\@�
=�.�R>�G�@ffC��\@�
=�-��?�=q@�z�C��H                                    Bx�w��  T          A>�R@���.=q>�\)?���C���@���-G�?�
=@�p�C��R                                    Bx�xj  "          A>ff@�33�/\)>u?�z�C�e@�33�.ff?�33@���C�s3                                    Bx�x  
�          A?33@�{�/\)?L��@tz�C���@�{�-?��HA�C��\                                    Bx�x#�  �          A?
=@��\�/�
?p��@��C�P�@��\�-�?�{A33C�l�                                    Bx�x2\  "          A>�\@�ff�0��>�{?�z�C��)@�ff�/�?��
@�p�C��                                    Bx�xA  "          A>ff@�33�2�R>\)?+�C��@�33�1�?��@���C�!H                                    Bx�xO�  T          A?�@Dz��9����#�
C�n@Dz��8��?\(�@�(�C�s3                                    Bx�x^N  �          A@��@��
�4�ÿ   �
=C��@��
�4��>��H@z�C��                                    Bx�xl�  T          AC33@\(��;33>��?5C�/\@\(��:=q?�@���C�9�                                    Bx�x{�  "          ABff@o\)�8�;�Q��Q�C���@o\)�8��?(��@G�C��q                                    Bx�x�@  �          AA�@dz��8Q��녿�C���@dz��8(�?�R@<(�C��)                                    Bx�x��  T          AA�@U��9p��
=�3�
C�f@U��9��>�ff@�C�                                    Bx�x��  
�          A?�
@W��7���R�>�RC�+�@W��7�
>��?�p�C�(�                                    Bx�x�2  T          A>{@G��5녿��H�ᙚC���@G��7\)��
=��C���                                    Bx�x��  �          A?�
@S�
�6ff�����ӅC��@S�
�7����ÿ���C��                                    Bx�x�~  
�          AAG�@Z�H�8��>�p�?޸RC�=q@Z�H�7�?�
=@ٙ�C�K�                                    Bx�x�$  �          A@��@S�
�8�ÿ=p��`��C���@S�
�9G�>�{?���C���                                    Bx�x��  �          A@(�@C33�9G��W
=����C�c�@C33�9��>u?�33C�`                                     Bx�x�p  "          AAG�@X���8  ����\)C�33@X���9G����ÿǮC�%                                    Bx�y  �          AAp�@h���6ff���H�C�ٚ@h���8(���R�;�C��                                    Bx�y�  �          AA��@g
=�6�\�����
C�Ǯ@g
=�8Q�0���QG�C���                                    Bx�y+b  "          A@(�@vff�2�H�ff� ��C�~�@vff�5G����\��33C�aH                                    Bx�y:  
�          A?33@}p��/�
���<  C���@}p��2�H��{����C�                                    Bx�yH�  
�          AA@��H�333����"�HC�@��H�5�����C��                                    Bx�yWT  T          AB=q@����4(����=qC�ٚ@����6�\�k����HC��)                                    Bx�ye�  "          AB{@{��4���z��G�C��R@{��7
=�s33��\)C�y�                                    Bx�yt�  
�          A?\)@j�H�2�H����$z�C�R@j�H�5p���G����C���                                    Bx�y�F  
�          A=�@r�\�1녿��H�z�C�g�@r�\�3�
�
=�5C�P�                                    Bx�y��  T          A<��@}p��1G��Tz�����C���@}p��1>�z�?�z�C�Ф                                    Bx�y��  
�          A<Q�@����0�þ�(���C���@����0��?333@X��C�H                                    Bx�y�8  
�          A:�R@�G��/
=����C�#�@�G��.�H?#�
@HQ�C�%                                    Bx�y��  �          A:{@w��/
=�=p��i��C���@w��/\)>Ǯ?�z�C���                                    Bx�ȳ  	�          A:�\@r�\�/�
�O\)�~{C��@r�\�0(�>���?��C��                                     Bx�y�*  
�          A;33@xQ��0(��B�\�n{C��{@xQ��0z�>Ǯ?�C���                                    Bx�y��  T          A;33@y���0  �.{�S�
C�� @y���0(�>�@ffC���                                    Bx�y�v  T          A;
=@mp��0�ÿ
=�:=qC�G�@mp��0��?z�@5�C�G�                                    Bx�z  
�          A<z�@p���1p���p���  C�]q@p���2�\�L�;k�C�O\                                    Bx�z�  �          A>{@z=q�1���R��ffC���@z=q�3\)��zῴz�C���                                    Bx�z$h  �          A=p�@}p��0z���H��C��@}p��2ff���!G�C��=                                    Bx�z3  
�          A<��@z=q�/�
�����RC��@z=q�2{�.{�Tz�C���                                    Bx�zA�  T          A<z�@{��.�\�ff�#�
C��f@{��1��fff���C���                                    Bx�zPZ  T          A<z�@�G��-p��
�H�*=qC�8R@�G��0Q�xQ����C�3                                    Bx�z_   �          A;�@e�0(���=q��G�C�
=@e�1p���G���C���                                    Bx�zm�  "          A9@>�R�3
=�333�\��C�s3@>�R�333?��@,��C�q�                                    Bx�z|L  
Z          A;\)@6ff�4(���(���RC�
@6ff�5���W
=��  C��                                    Bx�z��  T          A:�H@AG��1����\� ��C���@AG��4(��G��tz�C��H                                    Bx�z��  T          A;
=@C�
�0���\)�0z�C���@C�
�3�
�xQ�����C��)                                    Bx�z�>  
�          A9@:=q�0Q��
=q�+\)C�ff@:=q�333�c�
����C�J=                                    Bx�z��  �          A8��@<(��.�H��\�7\)C���@<(��1녿��
���
C�h�                                    Bx�zŊ  T          A8Q�@>{�,���)���U�C��\@>{�0�׿����ffC���                                    Bx�z�0  "          A6�H@N{�+
=�(��D��C�c�@N{�.ff��
=��ffC�<)                                    Bx�z��  T          A6�R@L(��*�R�!G��L(�C�S3@L(��.=q��G���33C�*=                                    Bx�z�|  T          A7�@O\)�,  ���>�\C�e@O\)�/33������RC�@                                     Bx�{ "  
�          A7�@\(��+33�G��6�\C��@\(��.=q�}p���
=C��f                                    Bx�{�  
�          A6�H@S33�,Q��z����C��@S33�.�R��R�FffC�h�                                    Bx�{n  
�          A8z�@i���.�\��\�#33C�B�@i���.=q?J=q@|(�C�Ff                                    Bx�{,  {          A9G�@|���-�8Q��c�
C��q@|���-�?
=@:=qC��)                                    Bx�{:�  -          A8  @s�
�-p�=L��>��C���@s�
�,  ?�{@�G�C�                                    Bx�{I`  �          A7�
@k��.{���#�
C�Z�@k��,��?��@�C�j=                                    Bx�{X  T          A7�
@l(��-�O\)���C�aH@l(��-�?�@$z�C�]q                                    Bx�{f�  
�          A7\)@qG��+���p���C���@qG��,��=�Q�>��C��q                                    Bx�{uR  T          A5��@����)p�>�G�@�RC�e@����'\)?޸RA��C��                                    Bx�{��  "          A6ff@�G��(z�:�H�j�HC�3@�G��(��?z�@;�C��                                    Bx�{��  
�          A6�R@�G��(�Ϳh����=qC�\@�G��)G�>��@z�C��                                    Bx�{�D  �          A733@�Q��)p��^�R���HC��3@�Q��)�>��@C��                                    Bx�{��  "          A5��@��&�\����C��@��(�׾�{�޸RC���                                    Bx�{��  T          A4��@�=q�%녿����
=C��R@�=q�(z��R�I��C���                                    Bx�{�6  T          A4��@x���'\)��{��
C�4{@x���)���'
=C��                                    Bx�{��  
�          A733@XQ��,�Ϳ�G����C��3@XQ��/
=��Q��C���                                    Bx�{�  
�          A6ff@Vff�+�
���z�C��=@Vff�.=q��G����C��                                    Bx�{�(  T          A4  @�=q�&{��ff��z�C���@�=q�'�=u>���C��q                                    Bx�|�  T          A2ff@���$�Ϳ��
��=qC���@���&{=��
>���C���                                    Bx�|t  �          A2=q@����$z῰����(�C�� @����%논��
��C���                                    Bx�|%  �          A3�
@�p��%�����\�ϮC��{@�p��&�H=�G�?
=qC��                                    Bx�|3�  T          A2�\@�Q��%��������p�C��R@�Q��&�\>�\)?�
=C���                                    Bx�|Bf  �          A2ff@|(��%녿����33C�e@|(��&�H>�  ?��C�W
                                    Bx�|Q  T          A1@l���&=q��=q��(�C���@l���'�=�\)>�p�C���                                    Bx�|_�  
�          A0��@}p��$(���G����C��\@}p��$��>\?�p�C��                                    Bx�|nX  "          A0z�@vff�$�׿xQ����C�E@vff�%�>�(�@{C�<)                                    Bx�||�  "          A/�
@w
=�#���ff��\)C�T{@w
=�$z�>�33?�C�H�                                    Bx�|��  "          A/33@xQ��#\)�=p��x��C�e@xQ��#�?(��@\��C�c�                                    Bx�|�J  �          A0��@|(��$�׿0���eC�z�@|(��$��?:�H@r�\C�z�                                    Bx�|��  "          A0z�@|(��$�׿   �&ffC�z�@|(��$(�?n{@��C���                                    Bx�|��  �          A0(�@tz��$�þ\�   C�,�@tz��$Q�?�ff@�\)C�7
                                    Bx�|�<  �          A0z�@tz��%G�>k�?�(�C�'�@tz��#33?�A�C�Ff                                    Bx�|��  �          A/33@}p��"�R��G����C��f@}p��!G�?���@�ffC���                                    Bx�|�  "          A.�R@i���$Q�!G��S33C��@i���$(�?Tz�@��HC���                                    Bx�|�.  T          A.=q@z=q�!녿aG���z�C��\@z=q�"=q?\)@=p�C��=                                    Bx�} �  
�          A-G�@u� �ÿ���
=C�p�@u�!�>�=q?�
=C�aH                                    Bx�}z  �          A-@s33�!p�������C�U�@s33�"�R>.{?c�
C�B�                                    Bx�}   �          A,��@y��� Q쿕��\)C���@y���!G�>�\)?�p�C��3                                    Bx�},�  �          A,  @xQ���������{C���@xQ�� Q�>Ǯ@33C���                                    Bx�};l  "          A,(�@�  �\)�J=q��{C��{@�  ��?+�@c�
C���                                    Bx�}J  
�          A*�\@u���R�E���z�C��@u���R?0��@l(�C���                                    Bx�}X�  
�          A*�\@w��=q�fff��33C��\@w���\?\)@@��C���                                    Bx�}g^  �          A+
=@x���녿�G���Q�C�� @x���33>W
=?���C���                                    Bx�}v  
�          A-��@g
=�"=q�h�����C��@g
=�"�\?(�@Mp�C���                                    Bx�}��  �          A0��@/\)�+33>�{?�  C�/\@/\)�(��?�
=A z�C�K�                                    Bx�}�P  �          A1�@*=q�+�
>u?�G�C��3@*=q�)G�?���A��C��                                    Bx�}��  T          A1G�@0  �+����\)C�1�@0  �)?˅A�C�E                                    Bx�}��  
�          A0Q�@�H�,  =L��>uC�U�@�H�)�?�A\)C�h�                                    Bx�}�B  "          A/\)@)���)����z��G�C��@)���(Q�?���@�ffC�{                                    Bx�}��  
(          A.�R@dz��$z�xQ���=qC���@dz��$��?(�@Mp�C���                                    Bx�}܎  	�          A,z�@8Q��%녾��p�C�@8Q��%�?���@���C�˅                                    Bx�}�4  �          A+�@%��&�\�L�Ϳ�=qC���@%��%�?�33@�Q�C�                                    Bx�}��  �          A+\)@���'��W
=����C��@���&=q?�@��HC���                                    Bx�~�  �          A*=q@#33�%G����R��z�C��\@#33�$  ?�ff@�Q�C��)                                    Bx�~&  �          A*�\@#33�%p����R��C��@#33�$(�?��@�G�C���                                    Bx�~%�  "          A+33@*�H�%�����
��p�C�:�@*�H�$z�?�ff@߮C�G�                                    Bx�~4r  �          A+�@'
=�&ff���R���C�
=@'
=�%�?�=q@�(�C�
                                    Bx�~C  �          A+
=@<���#�����N�RC��@<���#33?��\@�Q�C��                                    Bx�~Q�  �          A/�
@�33��Ǯ�=qC�~�@�33��<�>��C�\)                                    Bx�~`d  �          A/�@�  �"�\��\)��(�C��q@�  �#\)?�\@*=qC���                                    Bx�~o
  T          A-�@|��� �ÿ�=q��\)C���@|���!?
=q@7
=C��=                                    Bx�~}�  �          A,��@�����H���H��  C��@���� ��>#�
?W
=C���                                    Bx�~�V  �          A+\)@n{��������\C�9�@n{� z�?
=q@8��C�/\                                    Bx�~��  "          A*=q@z�H��Ϳ��H�У�C���@z�H��>���@
=qC��
                                    Bx�~��  
�          A)p�@��
����� Q�C�ff@��
�=q��=q��p�C�4{                                    Bx�~�H  "          A((�@�=q�ff�����!�C�Y�@�=q����\)���C�&f                                    Bx�~��  �          A'�
@����� ���0��C�H�@�����þ�G���HC�                                    Bx�~Ք  "          A'�@���z�����R�HC�G�@����ͿQ���Q�C���                                    Bx�~�:  �          A(��@��\�z���
�K33C���@��\��׿=p�����C�7
                                    Bx�~��  "          A(  @���
�޸R���C�k�@��ff�8Q�z�HC�8R                                    Bx��  "          A'�@����H�����,  C�t{@���{�Ǯ���C�5�                                    Bx�,  
�          A)��@�
=����
=�NffC�ff@�
=��ͿO\)��z�C��                                    Bx��  -          A,  @���(����K\)C��
@���z�E���33C��H                                    Bx�-x  �          A+
=@���G�����HC�<)@���  �B�\���
C�
=                                    Bx�<  	�          A(��@�33�p���G��\)C���@�33�\)>��?O\)C�q�                                    Bx�J�  T          A'�@�=q��ÿ�����p�C��=@�=q�ff>���?��C�q�                                    Bx�Yj  �          A'�
@{��ff��
=��\)C��@{��\)>��H@+�C�f                                    Bx�h  �          A(��@mp��G��k���  C�T{@mp��p�?J=q@�G�C�P�                                    Bx�v�  
�          A'�@l(��(��Y����z�C�Y�@l(��(�?\(�@�{C�Y�                                    Bx��\  	�          A((�@^�R�녿Tz���G�C��R@^�R��?fff@���C���                                    Bx��  
Z          A*{@P���!���z��HQ�C��@P��� ��?�Q�@�C��)                                    Bx���  
�          A+�@C33�$z����RC�AH@C33�#33?��@�{C�P�                                    Bx��N  "          A+
=@-p��$�ÿ:�H�x��C�Z�@-p��$z�?���@���C�`                                     Bx���  
�          A+\)@.�R�%p����C�
C�c�@.�R�$Q�?�G�@أ�C�n                                    Bx�Κ  "          A+�
@@���$�Ϳ��1G�C�'�@@���#�?��@���C�5�                                    Bx��@  �          A+�
@P  �#����p�C���@P  �"=q?�\)@��C��                                    Bx���  T          A+�@P  �#33�
=q�7�C���@P  �"=q?�ff@�{C���                                    Bx���  "          A+�@X���"=q��R�Tz�C�AH@X���!p�?��H@�\)C�L�                                    Bx��	2  �          A+�@\���!�O\)��=qC�l�@\���!��?��
@�  C�p�                                    Bx���  
�          A+33@S�
�"=q�^�R��p�C�
=@S�
�"{?z�H@��C��                                    Bx��&~  �          A,  @Z�H�"ff�aG���{C�Q�@Z�H�"=q?z�H@�\)C�T{                                    Bx��5$  "          A+�@_\)�!�h����33C��@_\)�!��?s33@��\C��                                    Bx��C�  
�          A+\)@j=q� (���\)��\)C��@j=q� ��?:�H@{�C�H                                    Bx��Rp  T          A*=q@P���!���8Q��x��C��@P��� ��?��@�C���                                    Bx��a  �          A)�@[�� z�5�w
=C�p�@[���
?�33@�{C�y�                                    Bx��o�  
�          A)��@P  � �ÿJ=q��Q�C��{@P  � z�?��@�z�C���                                    Bx��~b  T          A)G�@}p�����
=����C��@}p��Q�?&ff@aG�C��                                    Bx���  
�          A(��@���녿�����C�t{@���\)>�(�@ffC�\)                                    Bx����  "          A)�@u���׿�����C��=@u���?L��@�=qC���                                    Bx���T  T          A(Q�@l(��G��=p�����C�H�@l(����?�\)@��HC�Q�                                    Bx����  �          A(��@x����
���\���C��f@x���  ?Tz�@�G�C��H                                    Bx��Ǡ  T          A(��@x���  �u��  C�޸@x���(�?fff@�z�C��q                                    Bx���F  
�          A(  @~�R��R�}p���z�C�0�@~�R��H?^�R@��C�.                                    Bx����  �          A'
=@n�R�
=�O\)��\)C���@n�R��R?�ff@���C��                                    Bx���  "          A%G�@O\)��׿��K�C�&f@O\)�\)?��@�G�C�7
                                    Bx��8  
�          A$��@6ff���p���\C��@6ff�  ?��
A	p�C�'�                                    Bx���  "          A$��@G
=�  �xQ�����C���@G
=�  ?n{@�{C��{                                    Bx���  "          A%�@8���G�������(�C�0�@8���?Q�@�=qC�*=                                    Bx��.*  T          A'
=@Z=q�Q쿰�����HC��H@Z=q���?
=q@?\)C��                                    Bx��<�  
�          A&�R@n{��R��=q���RC��f@n{�
=?Q�@���C��                                     Bx��Kv  "          A&�H@y���녿n{���C�f@y����?xQ�@�33C�f                                    Bx��Z  "          A&�R@}p��p��xQ����
C�5�@}p��p�?n{@�33C�4{                                    Bx��h�  
�          A'\)@{��=q�xQ����\C��@{��=q?s33@�{C��                                    Bx��wh  "          A&�\@{��G������  C�*=@{����?^�R@���C�%                                    Bx���  �          A'�@����p���ff���C�s3@����?aG�@���C�o\                                    Bx����  T          A%p�@��\���z��O\)C���@��\��\?��@��C��R                                    Bx���Z  �          A$z�@��
��\���:�HC�Ф@��
�  ?�  A�C�                                      Bx���   
�          A#�
@�z���#�
�L��C��@�z���R?�\)A((�C�%                                    Bx����  �          A$Q�@�ff����ÿ�C�3@�ff��
?ǮA�
C�8R                                    Bx���L  "          A$  @�
=��H�@  ��\)C��)@�
=�=q?�{@ǮC��                                    Bx����  �          A#\)@�\)��R��ff��
C��)@�\)���>�z�?���C�Ф                                    Bx���  �          A#\)@���\�ٙ��C��q@���>\)?J=qC���                                    Bx���>  "          A"ff@�\)�Q쿘Q���  C�4{@�\)�G�?+�@s33C�"�                                    Bx��	�  "          A"=q@�����ÿ���C���@�����\>�@'
=C���                                    Bx���  �          A ��@���33������
C��q@���G�>���?ٙ�C��3                                    Bx��'0  �          A (�@������@(�C���@������W
=��(�C�E                                    Bx��5�  "          Aff@�G�����Z�HC�k�@�G��z���H�4z�C�\                                    Bx��D|  �          A=q@|�������B{C�@|����þL�Ϳ�
=C��f                                    Bx��S"  
�          A33@p����׿�p��   C�N@p���33>B�\?���C�!H                                    Bx��a�  _          A"=q@|������
=�?�
C���@|���p��#�
�h��C�t{                                    Bx��pn  T          A"{@~�R�z��{�J�HC��@~�R��;�\)�˅C��R                                    Bx��  �          A ��@�{�z�� ���8��C�ff@�{�(����333C�)                                    Bx����  T          A!�@��\�  �����ۅC���@��\���?8Q�@��
C��=                                    Bx���`  �          A33@�=q�(��s33���C�%@�=q�(�?z�H@�z�C�&f                                    Bx���  
�          A33@�33�녿�
=���HC�@�33��R?8Q�@�p�C��q                                    Bx����  T          A
=@����
�H�:�H��ffC�j=@����
{?��@�=qC�y�                                    Bx���R  T          Aff@�  �=q��
=���C��@�  �z�?���@��C�0�                                    Bx����  T          A
=@�����
�\)�L��C��\@�����R?�(�@��C�                                    Bx���  
�          Aff@�Q���녿���]p�C��@�Q���  ?��@��
C��f                                    Bx���D  �          Aff@�(���  �B�\��z�C�33@�(���?aG�@�33C�9�                                    Bx���  
�          A�\@����
��
=��HC���@���{?�\)@��RC�                                      Bx���  T          A\)@�(���H�&ff�qG�C�E@�(��{?��@љ�C�Z�                                    Bx�� 6  �          A33@����H�Q����C�9�@����\?z�H@�(�C�@                                     Bx��.�  T          A�H@�{�p��s33��Q�C��@�{���?Tz�@��C���                                    Bx��=�  "          A�H@��\��R�\(����C�(�@��\��\?s33@��RC�,�                                    Bx��L(  �          A�H@�  ���H��  ����C��
@�  ���
?@  @��C��=                                    Bx��Z�  �          A�H@\��=q�0���~�RC��\@\����?�ff@�=qC��                                    Bx��it  �          A�\@����R�.{�z�HC�\)@���p�?��@�G�C�o\                                    Bx��x  �          A@���=q�O\)��\)C�  @����?�G�@�z�C��                                    Bx����  T          A=q@�ff� �Ϳ(���s�
C���@�ff���?�33@��C��                                     Bx���f  �          A@�� (��L����p�C���@����?�  @��HC���                                    Bx���  "          A��@���� z�=p���=qC���@������?���@�  C��                                     Bx����  
�          Ap�@�
=��\)�\)�R�\C��{@�
=����?��R@�\)C���                                    Bx���X  T          Ap�@�
=�=q�@  ��33C��3@�
=���?���@�{C��                                    Bx����  �          A(�@�z���ff�aG����C���@�z���ff?aG�@��RC���                                    Bx��ޤ  
Z          A�R@����   ���޸RC��\@���� ��?+�@~�RC��
                                    Bx���J  "          A=q@��R���s33��z�C�� @��R��?�  @�
=C��H                                    Bx����  �          A33@�G��
�R�z��Z�HC�+�@�G��	�?�z�AC�K�                                    Bx��
�  "          A�R@����	�G����
C�C�@����	�?��H@�C�U�                                    Bx��<  
Z          A�\@��
�33�(��h��C�Q�@��
��?��@�
=C�p�                                    Bx��'�  
�          A�\@�(��33��
=��RC�]q@�(���?��
A=qC��                                    Bx��6�  �          A�H@�������EC�B�@����?�z�AC�k�                                    Bx��E.  T          A
=@��\����\�B�\C�ٚ@��\��?�33A��C��                                    Bx��S�  �          A  @���\)������
C���@����?s33@�=qC���                                    Bx��bz  
�          A�R@�  � zῂ�\��G�C��3@�  � ��?aG�@��RC���                                    Bx��q   �          AQ�@��
��������ffC��3@��
���R?:�H@���C���                                    Bx���  �          A  @����p����
��G�C���@����Q�?��@N{C���                                    Bx���l  
�          A\)@�{��z�У����C�>�@�{��33����c�
C��                                    Bx���  �          A33@������׿����C�*=@������>\)?Q�C�ٚ                                    Bx����  �          A z�@�\)��\���R��C�N@�\)��p�>�@'�C��                                    Bx���^  "          A ��@�z��߮�������C��\@�z����?
=q@C�
C���                                    Bx���  �          A�@�(����c�
��z�C��f@�(���?Tz�@�Q�C��                                    Bx��ת  �          A�
@��
�߮��\)��{C�@��
�ۅ?���@�
=C��                                    Bx���P  �          A33@����
>L��?�33C�'�@���(�?�\A#�C���                                    Bx����  T          A�H@�����þ��
��\)C�H@����z�?�AC�E                                    Bx���  �          A��@˅��\)�\)�P��C�0�@˅��z�?��R@��C�Y�                                    Bx��B  
�          Aff@���G�=�Q�?�C�,�@���=q?�(�A ��C��                                    Bx�� �  "          A=q@��R��  =L��>���C��@��R��=q?��RAz�C�|)                                    Bx��/�  T          A��@Ǯ��ff�}p����HC��R@Ǯ��
=?Tz�@��
C��\                                    Bx��>4  �          A33@�z���?���A  C���@�z���  @P��A�=qC��)                                    Bx��L�  �          A�@����  ����7�C��@����\)�#�
��Q�C���                                    Bx��[�  �          AQ�@�G�� Q�#�
�x��C�@�G���{?���A (�C�0�                                    Bx��j&  "          A��@��R��\)����fffC���@��R��z�?�{A33C���                                    Bx��x�  �          AQ�@������;\)�Q�C��@�����ff?�ffA.{C�E                                    Bx���r  
�          A(�@�����
=��  ��  C��=@�����?�33A   C��                                    Bx���  
Z          A=q@�Q���zᾨ����C��{@�Q���  ?��RAffC�:�                                    Bx����  T          A�R@�Q�����������C�R@�Q���Q�?���A�RC�g�                                    Bx���d  �          A�@�  ���Ϳ��
���C��=@�  ��?z�@`��C��                                     Bx���
  T          A  @�33�����  ���HC�w
@�33��p�?n{@�p�C�s3                                    Bx��а  T          A��@�G��  ��\)�6ffC���@�G��\)>k�?���C�q�                                    Bx���V  T          A��@�����(���  �(�C�� @����   ?   @E�C�l�                                    Bx����  T          Ap�@�=q� ���33�_�C��=@�=q�녾k���{C�P�                                    Bx����  �          A{@�����z�޸R�*�\C���@����G�>�z�?�G�C�G�                                    Bx��H  �          A  @��R���   �C
=C�e@��R��=�Q�?�C��                                    Bx���  �          A��@��������ff�-p�C�@����
=>�=q?���C�t{                                    Bx��(�  	`          A�@�  ��  �z��g
=C��R@�  ���H��Q��{C�B�                                    Bx��7:  
�          A�\@�{�����,(�����C�5�@�{��\)�Tz���\)C�]q                                    Bx��E�  
�          A��@����\)�e���p�C���@��������H�)�C���                                    Bx��T�  �          A@�{����dz���C�s3@�{���H���H�(z�C�.                                    Bx��c,  �          A��@�
=��z�� ���C
=C��@�
=�=q=u>���C���                                    Bx��q�  �          A33@�{��=q�|(����C��)@�{���
����c�
C�^�                                    Bx���x  �          AQ�@�p���Q��5����RC��@�p���  �s33��ffC�C�                                    Bx���  "          A�R@�����H��z���z�C��
@����ff� ����p�C�                                    Bx����  �          Az�@��������{��33C���@�������C33��
=C�H�                                    Bx���j  �          A�@�����G���G��C�  @�����p�?
=q@O\)C���                                    Bx���  �          A��@������  ��{C��@����?\(�@�G�C��                                    Bx��ɶ  �          A  @�33��H�k�����C���@�33�33?���A>{C���                                    Bx���\  �          A�H@��
�p���\)��G�C��{@��
�G�@AL��C�7
                                    Bx���  T          AG�@�����\=L��>���C�8R@����@�RAW\)C���                                    Bx����  
�          A(�@�  ���?�p�A33C�>�@�  ���@mp�A�z�C�%                                    Bx��N  !          A\)@����
=q?xQ�@�Q�C�+�@����p�@L(�A��C��\                                    Bx���  �          A33@��	��?��@N�RC��)@���\@1G�A���C�:�                                    Bx��!�  �          A�@�z��z�?+�@|��C�Ff@�z�� ��@8Q�A��C��R                                    Bx��0@  T          A(�@����	�?�z�@ڏ\C��\@������R@W
=A�\)C���                                    Bx��>�  "          A(�@�ff���?�33Az�C��)@�ff��z�@fffA��C���                                    Bx��M�  �          A(�@�Q����?�Ap�C��@�Q����@g
=A��
C��=                                    Bx��\2  �          A�
@����	G�?�z�Ap�C�@ @������H@w�A�\)C�P�                                    Bx��j�  �          A�H@�=q�	?��HA#
=C��f@�=q��33@{�A��C��3                                    Bx��y~  T          A\)@���	G�?�{A��C�%@����33@tz�A��C�/\                                    Bx���$  �          A�R@�G���H?��@���C�#�@�G�����@`  A�C��                                    Bx����  T          A�\@����
=?�G�@�  C��@�������@]p�A��C��                                    Bx���p  "          A�\@�33��R?�33@ڏ\C�U�@�33����@VffA�ffC�B�                                    Bx���  �          A=q@��
�=q?��@ϮC�k�@��
��G�@R�\A��C�T{                                    Bx��¼  
�          A�@�=q�ff?��@ʏ\C�E@�=q���@QG�A�
=C�(�                                    Bx���b  �          A�@����=q?�@�  C�>�@�������@XQ�A��\C�/\                                    Bx���  �          A{@�(��\)?��A��C��@�(�����@g
=A�Q�C���                                    Bx���  �          A@�
=��R?�{@���C��)@�
=���@UA��HC��                                    Bx���T  �          A@����?��@�=qC���@����(�@S�
A���C��                                     Bx���  |          Ap�@����H?���A�C���@����  @e�A���C��                                    Bx���  
�          A��@�z��ff?��RA�HC�Ǯ@�z���{@mp�A�  C���                                    Bx��)F  �          A��@�G��p�?�\)A4z�C���@�G����@��A�G�C��R                                    Bx��7�  T          A�R@���@G�A@z�C���@����  @��RA���C�8R                                    Bx��F�  �          A�\@�z��@�AB�\C���@�z���@��A�(�C�,�                                    Bx��U8  
�          Aff@���G�@z�AF=qC���@����ff@���A��C�T{                                    Bx��c�  |          A{@�ff��?���A;
=C��@�ff��
=@���A�ffC�c�                                    Bx��r�  �          A��@�z���@��AO
=C�\@�z���=q@��A�z�C�}q                                    Bx���*  T          A�@�Q��p�@��AS
=C��=@�Q����@���A��
C��3                                    Bx����  
�          A��@�����@ ��AB=qC���@����{@�
=A��
C��R                                    Bx���v  �          A��@}p��  ?�=qA0��C�~�@}p���p�@��A˅C��=                                    Bx���  "          A��@�Q���ÿ=p����C���@�Q���H?�33A z�C���                                    Bx����  T          A33@�����L����(�C�˅@����?�=qA�C��                                    Bx���h  
�          A�@���  �����p�C�z�@��� ��?�33A9C���                                    Bx���  "          A�R@��� ��=�\)>�G�C��{@�����R@�
Ac
=C�                                      Bx���  �          A{@�33�p�?��
@��C��@�33��\)@S�
A��
C���                                    Bx���Z  
�          A{@s33��?�{@��HC�R@s33��=q@[�A�Q�C��
                                    Bx��   �          Aff@s�
�z�?^�R@��C�f@s�
��ff@N{A�z�C�˅                                    Bx���  "          A  @g
=�	�?�  Ap�C�XR@g
=���@vffA�  C�S3                                    Bx��"L  "          A��@U���?�{A�HC�T{@U�@r�\A���C�.                                    Bx��0�  �          AG�@S33�{?�Q�A
=qC�/\@S33�@w�A��HC�                                    Bx��?�  T          A��@I����?��@�  C���@I����@a�A���C�`                                     Bx��N>  �          A��@?\)���?u@�\)C�!H@?\)��H@^�RA��C��                                    Bx��\�  "          Ap�@4z��=q?�R@l��C���@4z��	p�@J�HA���C�3                                    Bx��k�  �          AQ�@   ���>8Q�?��C��@   �@333A�33C�Y�                                    Bx��z0  �          A��@/\)�ff>���?���C�Q�@/\)�
�H@:=qA���C���                                    Bx����  �          A��@�R��?�  @�Q�C��@�R���@e�A�(�C�O\                                    Bx���|  "          A�H@�\�=q?@  @�\)C��@�\�z�@X��A���C��f                                    Bx���"  �          Ap�?�=q�
=?�R@mp�C���?�=q��@Q�A�=qC�
=                                    Bx����  
Z          AQ�?����?Q�@�C���?��
�H@\(�A��C��f                                    Bx���n  �          A
=@�33��ff?�ffA&�HC���@�33�θR@b�\A�C��)                                    Bx���  �          Ap�@�33��33@z�AM�C�Ф@�33�tz�@S33A�C��                                    Bx���  �          A  @������@  A`��C��@������@j=qA���C�ff                                    Bx���`  "          A(�@߮��33@�AmG�C���@߮��@xQ�A�(�C��                                    Bx���  T          A�@�=q���@+�A��HC�J=@�=q��33@�A�\)C�H                                    Bx���  
�          Ap�@���Ǯ@%A���C���@����\)@���Aڣ�C�ff                                    Bx��R  	�          A��@ʏ\��
=@.{A�=qC��H@ʏ\��p�@���A�Q�C�b�                                    Bx��)�  �          A(�@�{��R@\)Az�\C���@�{��@�
=A�C��                                    Bx��8�  
�          A=q@�G���z�@<(�A�z�C��=@�G���  @��A�Q�C��R                                    Bx��GD  	�          A�@���Ϯ@AG�A���C�K�@�����H@���A�(�C�R                                    Bx��U�  T          AG�@��\�ʏ\@W�A�ffC��3@��\��=q@�=qB��C��)                                    Bx��d�  T          Ap�@�����ff@mp�A���C��@������@���BG�C���                                    Bx��s6  T          A�@�{��  @���A�33C�AH@�{�s�
@�  B�
C���                                    Bx����  �          A�H@�G���@��\Aٙ�C�޸@�G���
=@�z�B�C��H                                    Bx����  
�          A�\@|���޸R@uA�
=C���@|������@��RB {C�Ǯ                                    Bx���(  T          A�R@�
=���
@��HA�G�C�{@�
=��Q�@�B0�
C�b�                                    Bx����  "          A\)@�=q����@��HB�HC�ٚ@�=q���
@ʏ\B3�HC��
                                    Bx���t  �          Az�@����  @��B  C�q@������@θRB6z�C�,�                                    Bx���  "          A{@�z���
=@x��A�ffC��3@�z���G�@��
B�\C�T{                                    Bx����  
�          AG�@�  ��\)@Tz�A���C��f@�  ��{@�\)BffC�Z�                                    Bx���f  
�          Ap�@��\��=q@O\)A�=qC��{@��\���@��BffC�                                    Bx���  "          Az�@����p�@a�A�\)C��3@�����\@��HB�HC��                                    Bx���  T          A��@�Q���33@s33A���C�0�@�Q���p�@��B �C�Ff                                    Bx��X  "          A�
@��H��\)@r�\A͙�C���@��H����@��
B��C��\                                    Bx��"�  �          A�@�z���=q@`��A�  C���@�z���
=@�(�BffC���                                    Bx��1�  �          A��@|(��߮@c33A���C��{@|(����
@�
=B=qC��                                    Bx��@J  T          A�@O\)��@�A�\)C��@O\)���
@��B0
=C���                                    Bx��N�  �          A\)@W���(�@��Ar�\C��@W��Ӆ@�{A���C���                                    Bx��]�  T          AG�@>{�  ?p��@�33C���@>{���H@W
=A��C���                                    Bx��l<  
�          AG�@N{�{?�{@�\)C��\@N{��p�@_\)A���C���                                    Bx��z�  "          AQ�@C33��=q@\)Ao�C��f@C33��G�@�\)A��\C�R                                    Bx����  
�          A�
@(���ff��{���C��3@(���?�  A=qC��R                                    Bx���.  �          A�>���p�����QG�C���>����
��(���\C�                                      Bx����  �          A z�?&ff�����=q�C��C���?&ff��=q���R���\C��R                                    Bx���z  T          A�?����ff��{�1��C�ff?�������l������C�S3                                    Bx���   �          A�?��
��\)��\)�1G�C�G�?��
��{�o\)�ظRC��                                    Bx����  �          A�H?У���  ��G����C��\?У���\)�33�hz�C��R                                    Bx���l  "          @��?�{�Ϯ������C���?�{������|��C�7
                                    Bx���  T          @�p�?����Q���(��Q�C���?�����
�!G���G�C�7
                                    Bx����  �          A ��?�Q������ff�(�C�,�?�Q���33�  ���
C�Y�                                    Bx��^  �          A Q�?�z��Ϯ��Q�� �C�5�?�z���\)���q��C�,�                                    Bx��  �          @�\)?�ff��
=����{C�{?�ff���H�#33��33C��                                    Bx��*�  �          @�(�?������  �z�C��3?���
=�+���G�C���                                    Bx��9P            @��R?�
=��p���(�� �C���?�
=��ff�E���=qC��                                     Bx��G�  ,          @�ff?�=q��\)���H�(�C���?�=q��\�U��Ə\C�P�                                    Bx��V�  "          @�ff?��������{�-{C���?����ᙚ�\����(�C���                                    Bx��eB  
�          @�\)?ٙ���33����� ��C��?ٙ������G
=����C���                                    Bx��s�  �          A�@���(���ff��
C�4{@�����%��ffC��{                                    Bx����  �          @�{?����z���\)�C��f?����(��;�����C�xR                                    Bx���4  �          A�?�\)���������L�C�O\?�\)��  �������C���                                    Bx����  T          @��R?aG���{���K��C�3?aG���33����C�Ф                                    Bx����  �          @��?h����33��33�GffC�N?h�������=q� �\C��                                    Bx���&  "          @��R?�����z���G��N\)C�Ff?����ȣ���=q�33C�n                                    Bx����  "          @��?�p���Q����
�HQ�C�:�?�p��ʏ\����{C�z�                                    Bx���r  �          @���?�
=�����  �<
=C��?�
=��ff�l(���
=C�.                                    Bx���  "          @�
=?B�\������33�O�C�?B�\�Å����(�C��=                                    Bx����  �          @��Ǯ���H�����\�C�H��Ǯ��
=��33��RC�                                    Bx��d  �          @�\)����g����
�f��Cy�Ύ����H��{�!�RC�                                    Bx��
  �          @�녿J=q�L(���Q��|��C�\�J=q�����ff�5C���                                    Bx��#�  �          @�=q����*�H����
C�  �����
=��33�K�
C�'�                                    Bx��2V  �          @��
���{��Q�C�K������
��ff�Z�C��=                                    Bx��@�  �          @�z�L���@  �����C��L����G�����B�C�ٚ                                    Bx��O�  "          @����=q�E��\�C��׾�=q������Q��@��C��H                                    Bx��^H  �          @��z��%���R�qC���z���ff��G��O��C��                                     Bx��l�  �          @��    �:=q��Q��HC��    ���R��  �D�HC��                                    Bx��{�  
�          @�33?��\��(���  �_
=C��
?��\��(��������C��                                    Bx���:  
�          @�?���r�\��\)�i
=C�k�?����(����R�"��C�y�                                    Bx����  �          @�ff?
=�Z=q���
�}�C���?
=��z����R�4�C�                                      Bx����  �          @�\)?&ff�^{��(��{G�C�C�?&ff���R��{�2�
C�`                                     Bx���,  "          @�Q�>��Z=q��ff�~z�C��)>���p������5��C���                                    Bx����  �          @�G�?\)�S�
�����C��\?\)��33���
�8�C���                                    Bx���x  �          @��?z�H�XQ���ff�|p�C�q?z�H���������4�HC�*=                                    Bx���  �          @���?G��e���
�wp�C�
?G����\��z��.��C���                                    Bx����  
�          @��?
=�Tz������RC�?
=���
���
�8�\C�%                                    Bx���j  "          @�?.{�Tz���p�k�C��3?.{�����  �:�C���                                    Bx��  T          @��
?n{�\(�����|p�C���?n{��\)��=q�4G�C��=                                    Bx���  �          @�(�?����\����  �z�
C��{?��������G��3{C�~�                                    Bx��+\  T          @���?5�C33��G�C��
?5���
���R�?Q�C��R                                    Bx��:  
�          @��<#�
�5��  ��C�)<#�
�������E�RC��                                    Bx��H�  �          @�p�>L���Mp���ffL�C��H>L����  ��=q�:��C�3                                    Bx��WN  T          @��
��Q��P����(�Q�C�=q��Q���������8�\C���                                    Bx��e�  T          @�z���\(�����|C���������2��C�aH                                    Bx��t�  T          @�녾B�\�p���љ��pp�C���B�\������Q��&G�C��                                    Bx���@  T          @�(��L����33�Å�`ffC�� �L�����\��\)�(�C��                                    Bx����  
�          @�\)��33�g���\)�o  C�8R��33��p���Q��$�
C�#�                                    Bx����  �          @�R�L���aG��ȣ��r��C�h��L�����H��=q�(=qC��3                                    Bx���2  
�          @�=q>�=q�����Q��M=qC��)>�=q��{�s�
���C�J=                                    Bx����  T          @�Q콣�
��  ��33�G�RC��H���
��\)�hQ���=qC��                                     Bx���~  "          @�Q�G��z=q���H�c33C�aH�G�����������C�
                                    Bx���$  
�          @�G�����dz���Q��m33C}� �����z���G��$p�C���                                    Bx����  T          @�  ��(��C33�θR�|33CxE��(����R��z��4�RC�)                                    Bx���p  �          @�(��\(��aG������q��C���\(���z����(\)C�s3                                    Bx��  
�          @�p��8Q��z�H�љ��j�C��Ϳ8Q������{� ��C�y�                                    Bx���  T          @�����R�\���
33C�J=�����=q��ff�6
=C�3                                    Bx��$b  �          A	녾���33�� =qC{:��������\�v�
C���                                    Bx��3  
�          A
ff���Ϳz�H�	p�¦�C�'������l(���=qk�C�>�                                    Bx��A�  "          A\)��Ϳ����	G� ��Cw�R���������{�x{C�R                                    Bx��PT  �          A	녿.{����� z�CrG��.{�}p����xC�"�                                    Bx��^�  "          AQ�O\)������
Cv
=�O\)��
=��  �k(�C���                                    Bx��m�  T          A
=�L�Ϳ�33�  �Ct\�L��������Q��o{C���                                    Bx��|F  �          A\)�G��������\)Cp�)�G���  ��p��u  C�z�                                    Bx����  
�          A�!G���{�
=(�Cx�Ϳ!G���p���\)�n��C���                                    Bx����  "          A{�.{�޸R��\=qCx�f�.{��G�����k�C�~�                                    Bx���8  "          A	��Y����
�z��fCwk��Y�����
����b�HC���                                    Bx����  �          A  ��G��	����Rp�Ct���G���������`��C��                                    Bx��ń  �          A��Y�����ff�
Cx�R�Y����G����`ffC��R                                    Bx���*  T          A�ÿ�33�����8RCt녿�33�������
�XQ�C��
                                    Bx����  	�          A�ÿ�=q�p��{z�CvO\��=q������(��XffC��R                                    Bx���v  	�          A33���� ���(�z�CwͿ������߮�V(�C��                                    Bx��   "          A
ff����(Q���\B�Cv�
�����z��ۅ�Q�C��                                     Bx���  "          A	���u�5� ��(�C{Y��u�����{�L�C��H                                    Bx��h  "          A	녿��H�:�H� (��Cw�\���H���
��(��I{C���                                    Bx��,  �          A	���У��<������
Cq
=�У����
�У��E(�C}{                                    Bx��:�  �          A�Ϳ�
=�.{��
=CnW
��
=����Ӆ�Jp�C{�R                                    Bx��IZ  �          A
=�   �Vff��\)�y�Co&f�   �����
=�4Q�CzY�                                    Bx��X   �          A����`  ��p��t��Cn�{����\)���
�/z�Cy��                                    Bx��f�  
�          A\)�33�_\)���u�Co�=�33��\)��(��0
=CzE                                    Bx��uL  �          A�R�!G��y����  �a�
Cm&f�!G���
=���H�  Cw&f                                    Bx����  T          @�=q���QG���\)�pG�Cm������G���G��+Cx�{                                    Bx����  �          AG�����Q���(��{z�Co������Q�����5�C{�                                    Bx���>  �          @�  ���
�I���޸R�}{Ct\���
��  ��G��5ffC}�q                                    Bx����  �          @��ÿ���/\)���
=Cp�3������������A\)C|�
                                    Bx����  "          A�׿���
=q�����RCo+������(���  �Z\)C~s3                                    Bx���0  "          Aff��=q����  \)Cf���=q�����{�]{Cz�                                    Bx����  �          @�׿�(���R�׮Cl����(���=q��(��L  C{@                                     Bx���|  T          @���@
=��=q����N�RC���@
=��
=>��H@�p�C��=                                    Bx���"  
�          @�ff@(����H��\)�Up�C�(�@(����>�@��RC��H                                    Bx���  
Z          @�{@   ���
������C��3@   ��Q�?��A@(�C�ٚ                                    Bx��n  "          @�  @Tz��hQ�@A�G�C�@ @Tz��*=q@Y��Bz�C��                                    Bx��%  �          @�p�@L(��\��@.�RA�33C�aH@L(��
=@mp�B,\)C��H                                    Bx��3�  4          @�=q@Fff�^�R@��AݮC��R@Fff��R@]p�B$(�C��=                                    Bx��B`  �          @�33@��
����@]p�A�  C�˅@��
�9��@�B&�HC�4{                                    Bx��Q  �          @�\)@�
=��\)@j=qA��
C��@�
=�n�R@�B"�C�ٚ                                    Bx��_�  �          A(�@����
@5A�33C�  @�����@�ffB	33C��                                    Bx��nR  �          Aff@��H��R@!G�A�\)C�>�@��H��
=@�  B  C���                                    Bx��|�  �          A
=@�33��R@��Aj�\C�  @�33�ə�@��HA�C�Q�                                    Bx����  �          AQ�@������
@���B=qC��f@����Z=q@���B:33C�@                                     Bx���D  �          A�@�33��33@��
A�G�C�AH@�33��\)@��
B2�HC�u�                                    Bx����  �          A��@��R�љ�@�=qA�  C��R@��R���H@�  B&�\C�R                                    Bx����  �          A��@�ff����@���A���C�J=@�ff��(�@�ffB�C��                                    Bx���6  �          A��@������@|��A��
C�XR@����z�@�\)B��C��                                     Bx����  �          A�@������@fffA�C��
@�����Q�@�\)B�RC���                                    Bx���  �          A��@�z�����@S�
A��HC��=@�z����@���B�C�h�                                    Bx���(  �          A{@�ff��\@{A{
=C�"�@�ff���
@�p�A�ffC���                                    Bx�� �  �          A�@�33��?p��@�
=C�<)@�33��ff@VffA��C��                                    Bx��t  �          A��@�����ff?Tz�@��C��@�����Q�@P  A�ffC�e                                    Bx��  T          AG�@�
=��?W
=@��\C���@�
=���H@S33A�ffC��q                                    Bx��,�  �          A(�@����
=?#�
@��HC�� @���ڏ\@E�A�{C��                                    Bx��;f  �          Ap�@��H�ָR@<��A�=qC��
@��H��33@�  Bp�C��f                                    Bx��J  �          A{@�����@�ffAۮC��@�����p�@�33B(  C�+�                                    Bx��X�  �          A�R@�p�����@|��A�ffC�>�@�p����R@�B!ffC�`                                     Bx��gX  T          A33@�=q��
=@���A���C���@�=q��  @���B$Q�C�                                      Bx��u�  �          Aff@���@W
=A�=qC���@���p�@��RB�
C�'�                                    Bx����  �          A�R@��R����@HQ�A��C���@��R���H@�G�B
=C��                                    Bx���J  �          A��@�����@EA���C�#�@����(�@�Q�Bp�C�@                                     Bx����  �          Ap�@�  ��Q�@7�A�33C��H@�  ���@���B�C��)                                    Bx����  �          A�H@�Q���\@A�A�Q�C��f@�Q���p�@��RB
Q�C���                                    Bx���<  �          A�H@��\��  @\��A���C�P�@��\��ff@��\BC��f                                    Bx����  �          A�\@��
��  @Tz�A�  C�h�@��
��  @�ffBffC�                                    Bx��܈  T          A�H@�33��{@|��A�z�C��
@�33���@�ffB"{C�3                                    Bx���.  
�          A�\@�z��љ�@�(�A�Q�C�c�@�z�����@�=qB&
=C��f                                    Bx����  �          A=q@�p���
=@L��A�ffC���@�p���Q�@��HBffC��                                     Bx��z  �          A@�(���G�@E�A�ffC�\)@�(����@�  B�HC�~�                                    Bx��   T          A{@�
=�ۅ@\(�A��HC��
@�
=��=q@���B��C��                                    Bx��%�  
�          A@�����
@`  A��\C��f@�����@��\B(�C�<)                                    Bx��4l  �          A{@�G����
@��RA�  C�h�@�G����
@�Q�B$�
C�7
                                    Bx��C  �          A{@����G�@��
A���C��@�����@�\)B#�C�C�                                    Bx��Q�  T          A{@��R��  @�A�=qC�'�@��R��Q�@�{B"z�C��                                    Bx��`^  �          A@�������@���A�p�C���@������@�
=BffC�K�                                    Bx��o  �          A{@����33@��RA�{C��f@���r�\@�=qB&�
C�{                                    Bx��}�  �          Ap�@�ff���@��A�p�C�!H@�ff����@��
B!33C��=                                    Bx���P  �          A�H@�{���@~{A�G�C�o\@�{���R@�ffB��C�'�                                    Bx����  �          AG�@����  @w�A�Q�C�j=@�����
@���B\)C���                                    Bx����  �          A@�����R@��A�G�C�Q�@����\)@���B!�RC�,�                                    Bx���B  �          A�\@�Q�����@��Aң�C��)@�Q���=q@�p�B!Q�C��f                                    Bx����  �          A��@�(����@��
A�ffC�K�@�(����@�=qB(
=C�y�                                    Bx��Վ  �          A33@����Q�@1G�A���C��@����ff@��
B\)C��R                                    Bx���4  �          A��@����
=@EA�=qC�#�@����=q@�=qB��C���                                    Bx����  �          A  @�\)��\)@H��A�
=C��@�\)��=q@�G�B=qC���                                    Bx���  �          A
�\@������@��A���C�AH@�������@��
A�RC�'�                                    Bx��&  �          A	��@��H����@�A�G�C�33@��H���@�(�A�G�C�#�                                    Bx���  �          A	��@�\)����@Mp�A��C�xR@�\)���@�G�B�C�K�                                    Bx��-r  �          AQ�@�\)���@{�A��HC��@�\)�~�R@�=qB$�C��\                                    Bx��<  �          A  @�(�����@��B�\C�~�@�(��"�\@θRBF=qC�:�                                    Bx��J�  �          AQ�@������@��HB  C�ff@����O\)@�Q�B4��C���                                    Bx��Yd  �          A33@�=q��(�@p��A�\)C�W
@�=q�tz�@��\BC�*=                                    Bx��h
  �          A33@����ff@W�A�z�C���@����Q�@��B=qC���                                    Bx��v�  �          A�@��\���R@AG�A���C��@��\��z�@��Bp�C��3                                    Bx���V  �          AG�@������@g�A�ffC�� @���p��@�p�BC�Z�                                    Bx����  �          AG�@�\)��\)@^�RA���C�z�@�\)���@�{B�RC���                                    Bx����  
�          A�@�G����Ϳ����
=C�˅@�G���z�?�{@��HC��                                    Bx���H  �          A�@�p���(��˅�0(�C��H@�p����?
=q@n�RC�W
                                    Bx����  �          A�R@�
=��\)�
=�i��C�:�@�
=��33    <��
C�c�                                    Bx��Δ  �          A��@�ff��{�����N�HC�B�@�ff��
=>u?У�C��)                                    Bx���:  �          A  @���ƸR��Q�� ��C��=@����\)?�p�AA��C�
                                    Bx����  �          A�H@��H���H?p��@�33C���@��H��@:=qA�\)C�G�                                    Bx����  �          @�
=@�����ff���Ϳ=p�C��f@�������?��A[�C��f                                    Bx��	,  �          @�G�@�\)��p�=��
?
=C�XR@�\)��=q?��HAk�
C�AH                                    Bx���  �          @�
=@�����?�
=A0��C��3@���33@N�RA�
=C���                                    Bx��&x  �          @�\)@�{����?�p�Amp�C�u�@�{���@s33A�C�q                                    Bx��5  �          @���@�\)����@z�A�Q�C���@�\)��p�@��A�  C���                                    Bx��C�  �          @�(�@�
=��(�@^�RAң�C�g�@�
=����@��B$Q�C��
                                    Bx��Rj  �          @�ff@{����R@o\)A��C�ٚ@{��z=q@�33B/�C���                                    Bx��a  �          @���@�������@�33A�G�C���@����g�@��
B8G�C��                                    Bx��o�  �          @�{@p  ����@��\B�HC���@p  �^{@\BG�C��
                                    Bx��~\  �          @�33@�z�����@�  BffC��R@�z��333@���BIG�C��3                                    Bx���  �          @�z�@x����@�  Bp�C��@x���Dz�@��
BL{C��
                                    Bx����  �          @�
=@��R����@��\B�
C��@��R�Mp�@��BB
=C�Y�                                    Bx���N  �          @�ff@�  ��(�@i��A���C���@�  ��33@�=qB(  C��                                    Bx����  �          A{@��R���þ8Q쿦ffC�*=@��R��\)?�A]��C��R                                    Bx��ǚ  �          A�
@��У׿��\��(�C��=@��Ϯ?���A=qC���                                    Bx���@  �          A��@��R��
=�����G�C�'�@��R��{��33�   C�
=                                    Bx����  �          A��@��H��Q�k���C�&f@��H���?��
AP��C��=                                    Bx���  �          A��@�\)�ʏ\@   A���C�b�@�\)��z�@��RBQ�C�O\                                    Bx��2  �          A�@e���@��A�Q�C�S3@e���H@�B9��C�Ф                                    Bx���  �          A��@��H��ff�3�
��Q�C��H@��H��=q�W
=���\C�S3                                    Bx��~  �          A�@�33������(���z�C�t{@�33����(Q���33C�!H                                    Bx��.$  �          A�@������\�dz���p�C��@����ƸR�����.=qC��3                                    Bx��<�  �          A�@��R�����Vff��ffC���@��R�Å��z����C��f                                    Bx��Kp  �          A�@����  �<�����RC�q@������u�ϮC�z�                                    Bx��Z  �          A(�@������P  ���C�Y�@�����
=������C�W
                                    Bx��h�  �          AQ�@�����=p����\C���@����33�}p��ָRC���                                    Bx��wb  �          AQ�@�������5����C�Z�@�����ff�k����C��)                                    Bx���  �          A�@�\)���\�Dz���=qC�q@�\)��=q��ff�p�C��                                    Bx����  �          A�
@�p���=q�7���=qC�O\@�p���\)���
��  C��\                                    Bx���T  �          A33@�(������:�H����C�T{@�(���ff�����p�C��f                                    Bx����  �          A��@�{�������J�HC��@�{���    �#�
C�9�                                    Bx����  �          AQ�@�p���{��G���Q�C�3@�p���ff?n{@�C�
=                                    Bx���F  �          A�@����ff�s33��(�C��@����{?}p�@�(�C���                                    Bx����  �          A  @����녿z�����C�G�@�����?���A"�\C��H                                    Bx���  �          A33@�����\)�0�����HC��q@�����z�?��RA�
C�ٚ                                    Bx���8  �          A�@�33��\)����w
=C���@�33���H?�\)A��C�&f                                    Bx��	�  �          A�
@�����\)��Q��RC��@�����{?�AK
=C��f                                    Bx���  �          A�@����{?�
=A<z�C�e@������@l��A�z�C���                                    Bx��'*  �          Aff@��
���H@��AtQ�C�@��
����@��A��HC��)                                    Bx��5�  �          A��@����У�?���A2�\C�t{@�����z�@g�AԸRC�q�                                    Bx��Dv  �          A�@��\�Ϯ?�A=�C��R@��\���\@mp�A�C��                                    Bx��S  �          A�\@����=q?�Q�A#33C��@����\)@aG�A��HC�ff                                    Bx��a�  �          A�@��H�˅?�ff@�  C���@��H���@E�A��
C�J=                                    Bx��ph  �          AG�@�G���p�?xQ�@���C�` @�G���  @A�A�{C��                                    Bx��  �          A (�@�����ff>�ff@P  C��=@������R@p�A��C���                                    Bx����  �          @��@�Q�����?!G�@�(�C���@�Q����@%�A��C�0�                                    Bx���Z  �          @���@�33����?
=q@|��C��@�33��Q�@$z�A�C�N                                    Bx���   �          @��\@��
���H>��@\(�C�U�@��
��33@(�A��C��                                    Bx����  �          @��H@�����\)>�33@%C��@�������@�\A�Q�C�"�                                    Bx���L  �          @�ff@�����z὏\)��\C���@�����=q?�Q�Ab�\C�j=                                    Bx����  �          @�(�@������=L��>ǮC�H@����@G�AnffC���                                    Bx���  �          @�33@�{��33=���?:�HC���@�{��\)@Av�HC�c�                                    Bx���>  �          @�(�@�
=�Å>�Q�@%�C��@�
=���@�A��C���                                    Bx���  �          @�(�@�G�����?�\@n�RC��\@�G�����@��A�G�C�(�                                    Bx���  �          @�=q@��
��=q?xQ�@���C��@��
��{@4z�A��C�T{                                    Bx�� 0  �          @��@��
��
=?�A
=qC��)@��
��Q�@C33A��\C�ff                                    Bx��.�  �          @�=q@�G�����?�ffA��C�AH@�G�����@L(�A�G�C�#�                                    Bx��=|  �          @��@�����(�@��A��RC��\@�����=q@{�A�ffC��                                    Bx��L"  �          @�
=@�Q���z�?�(�Ae��C��@�Q����R@g�A�z�C���                                    Bx��Z�  �          @�
=@����?���A6=qC��@���G�@S�
Aď\C�H�                                    Bx��in  �          A (�@�z���p�@  A���C���@�z����@s�
A�(�C���                                    Bx��x  �          A Q�@�p����H@��A��C�\)@�p��s33@uA�G�C���                                    Bx����  �          @�p�@�G�����@��A��C���@�G����@p  A�C��\                                    Bx���`  �          @�33@���=q?���AZ�HC�˅@���@^{A���C�c�                                    Bx���  �          @�(�@�������?�AXz�C�~�@�������@W�AʸRC�"�                                    Bx����  �          @��@�33���@Aw�
C�y�@�33�s�
@`��Aՙ�C�s3                                    Bx���R  �          @�z�@�\)���@�A��C�U�@�\)�c�
@k�A���C��q                                    Bx����  �          @��@�=q��
=@
=A�33C��@�=q�^{@k�A�=qC�!H                                    Bx��ޞ  �          @��@�Q���Q�@7�A��\C�ٚ@�Q��s�
@��\B�\C��                                    Bx���D  �          @���@�ff���\@_\)A��C�Q�@�ff�Mp�@�G�B33C�,�                                    Bx����  �          @��
@��\��\)@[�A�(�C��f@��\�9��@��B
=C���                                    Bx��
�  �          @�ff@�z���\)@)��A��
C��@�z��XQ�@}p�A�(�C���                                    Bx��6  �          @�
=@�p����\@$z�A�p�C���@�p��`  @z=qA�
=C�#�                                    Bx��'�  �          @�ff@�{��
=@)��A���C��@�{�XQ�@|��A�z�C���                                    Bx��6�  �          @��
@������@ffA��C�7
@����J�H@S33A��C��H                                    Bx��E(  �          @�  @��\�j�H?�=qA{�
C��R@��\�8��@:=qA���C���                                    Bx��S�  �          @�\)@��
�k�?�Ae��C�˅@��
�<��@0��A��
C��)                                    Bx��bt  �          @ָR@��H�l(�?ٙ�AjffC���@��H�<��@2�\AĸRC��                                    Bx��q  �          @�p�@��\�k�?�G�AQ�C��\@��\�@  @'
=A���C�N                                    Bx���  �          @��
@�G��g
=?�
=Ak
=C��3@�G��8��@0  A�(�C���                                    Bx���f  �          @ҏ\@�\)�j=q?�\)AdQ�C���@�\)�<��@-p�A�=qC�L�                                    Bx���  T          @љ�@�p��l��?���A]C�33@�p��@  @*�HA�z�C��=                                    Bx����  �          @��H@�{�p  ?ǮA[\)C��@�{�C33@+�A��C�Ǯ                                    Bx���X  �          @�(�@�Q��n�R?�ffAX(�C�P�@�Q��B�\@*=qA��C��R                                    Bx����  �          @Ӆ@����i��?�=qA]p�C��=@����<��@*=qA�  C�`                                     Bx��פ  �          @��
@��H�`  ?޸RAtQ�C�c�@��H�0��@1G�Ař�C�T{                                    Bx���J  �          @�Q�@�{�[�?���A�ffC�Ff@�{�*=q@5A��
C�k�                                    Bx����  �          @�=q@�
=�b�\?��A}p�C��@�
=�2�\@4z�A��C��3                                    Bx���  �          @�G�@����b�\?���A]��C�
=@����7
=@'
=A�(�C���                                    Bx��<  �          @�  @�z��l(�?��RAS33C�,�@�z��A�@%�A��HC��H                                    Bx�� �  �          @�ff@��R�[�?�=qAc33C�O\@��R�0  @%�A�p�C�\                                    Bx��/�  �          @�@��^�R?��
A\z�C�@��4z�@#33A�G�C��)                                    Bx��>.  �          @�ff@����mp�?�  AX��C��@����B�\@&ffA��HC�l�                                    Bx��L�  T          @�ff@�z��\(�?�{A�Q�C��@�z��*�H@6ffA�  C�AH                                    Bx��[z  �          @θR@�33�b�\?�ffA��C��H@�33�2�\@4z�A�
=C���                                    Bx��j   �          @У�@���{�?�33AF�RC���@���R�\@$z�A��C�N                                    Bx��x�  �          @�{@��H�g�?�G�AtQ�C���@��H�8Q�@3�
A���C��
                                    Bx���l  �          @أ�@�{�dz�?�=qA{33C�]q@�{�3�
@7
=A�(�C�T{                                    Bx���  �          @�  @��R�g�?��Aa�C�:�@��R�:�H@,(�A�(�C��                                    Bx����  �          @أ�@�G��tz�?���Ay�C�
@�G��C33@:�HẠ�C�                                      Bx���^  �          @�G�@��H�tz�?�(�Ak�C�9�@��H�E�@5�A�p�C�                                      Bx���  �          @�G�@����vff?�\)A�
C��3@����Dz�@>�RA�Q�C���                                    Bx��Ъ  �          @�  @�G��q�?�ffAw\)C�@ @�G��AG�@8��A���C�!H                                    Bx���P  �          @أ�@���p  ?�p�AmC���@���AG�@4z�A���C�K�                                    Bx����  �          @�
=@�33�k�?�  Aqp�C���@�33�<��@3�
A�C��                                    Bx����  �          @�
=@���g�?�Az�RC��)@���8Q�@6ffA���C��f                                    Bx��B  �          @ָR@��\�c33@G�A�Q�C�.@��\�/\)@A�A�Q�C�e                                    Bx���  �          @�\)@��
�E@�A��C��{@��
��\@<(�AΣ�C��                                    Bx��(�  �          @�
=@�  ��@FffAޣ�C�@ @�  ���@S�
A�33C�AH                                    Bx��74  �          @׮@�p��8Q�@aG�A�G�C�\)@�p�>���@dz�B �@��\                                    Bx��E�  �          @׮@�����@1�A�G�C��q@��Ϳ��@XQ�A��C��                                    Bx��T�  �          @ٙ�@�p���R@3�
AÅC�0�@�p����R@\(�A�\)C��f                                    Bx��c&  �          @�=q@��&ff@0��A��\C���@���\)@Z�HA��HC��                                    Bx��q�  
�          @�=q@�
=�8Q�@A��RC��f@�
=�G�@G�AمC�L�                                    Bx���r  
�          @��@�p��G�@
=A��\C���@�p��z�@>{A��HC��                                    Bx���  T          @ٙ�@���E@	��A�p�C���@����@@  A�G�C�f                                    Bx����  
�          @�ff@��\�=p�@��A��
C�  @��\���@@��A��
C�|)                                    Bx���d  �          @��@�Q��3�
@   A�=qC�xR@�Q��z�@O\)A��HC�p�                                    Bx���
  �          @�{@����{@4z�A�=qC��@��׿��R@\(�A���C�u�                                    Bx��ɰ  �          @׮@���\)@?\)A�p�C��@������@g
=B�C��
                                    Bx���V  �          @�  @��H���@8��Aʣ�C�%@��H����@`  A��
C���                                    Bx����  �          @׮@��H�#�
@/\)A�Q�C��\@��H����@X��A�(�C�                                      Bx����  �          @�\)@����*=q@0��A�=qC�%@��׿�Q�@\(�A�z�C�~�                                    Bx��H  �          @�  @�\)�;�@'�A���C���@�\)���R@X��A�33C�                                    Bx���  �          @ָR@�ff�S33@�A��C�j=@�ff�   @AG�A��C���                                    Bx��!�  �          @�ff@���N{@�
A�
=C��)@����@K�A�=qC�,�                                    Bx��0:  �          @�\)@����U�@�RA�G�C�,�@����\)@H��Aޏ\C���                                    Bx��>�  T          @�\)@����O\)@�A��C���@�����@O\)A�ffC�&f                                    Bx��M�  �          @أ�@�
=�J�H@=qA��HC��
@�
=��\@P  A�  C���                                    Bx��\,  �          @ٙ�@�\)�@��@+�A�  C��q@�\)��
@^{A�33C���                                    Bx��j�  �          @�  @���7
=@+�A��C�@ @����@Z=qA�\)C�`                                     Bx��yx  �          @���@��2�\@(Q�A�=qC�k�@���\)@UA�C��                                    Bx���  �          @��@�G��<��@�RA��HC��)@�G����@AG�Aי�C�o\                                    Bx����  �          @Ӆ@����<��@
=A��C���@����
=@H��A�p�C�U�                                    Bx���j  T          @Ӆ@�
=�HQ�?��HA�z�C�)@�
=���@3�
AɮC�*=                                    Bx���  �          @љ�@�p��Mp�?��
A|��C��3@�p��!�@*=qA��
C���                                    Bx��¶  �          @ҏ\@��R�H��?�33A�  C�@��R�(�@0  Ař�C���                                    Bx���\  �          @Ӆ@���G�?�Q�A��\C�8R@�����@1�A��C�8R                                    Bx���  �          @�\)@�Q��[�?�z�Ae�C�
=@�Q��1�@'
=A�\)C��)                                    Bx���  �          @�
=@��H�vff?��RA*ffC��@��H�S�
@�A��\C�q                                    Bx���N  �          @�Q�@����qG�?�z�AAG�C���@����K�@p�A�C��H                                    Bx���  �          @���@���XQ�?�\Aq�C�y�@���,��@,(�A���C�%                                    Bx���  �          @أ�@����Y��?�\)A���C�:�@����,(�@333A�p�C�
=                                    Bx��)@  �          @�\)@����QG�@G�A��C��3@����!�@9��AˮC��
                                    Bx��7�  T          @�p�@���S�
@�
A��C�E@���#�
@<��Aљ�C�\)                                    Bx��F�  �          @���@��
�@  @z�A�=qC��3@��
��@FffA��
C�b�                                    Bx��U2  �          @ٙ�@�{�:�H@33A�{C�ff@�{��@C�
A�C��=                                    Bx��c�  �          @�Q�@�ff�8��@�RA�(�C��@�ff�
=@>�RA�33C���                                    Bx��r~  �          @�
=@�{�1�@\)A�C���@�{�   @=p�A���C�L�                                    Bx���$  �          @�G�@����*=q@�A�p�C���@��ÿ���@C33A�33C�q                                    Bx����  �          @�=q@���-p�@��A��\C��
@������@8��A���C���                                    Bx���p  �          @ٙ�@�\)�,��@��A�G�C�g�@�\)��{@HQ�AۮC���                                    Bx���  �          @��@�Q��7�@{A�{C��H@�Q��ff@=p�A�  C��)                                    Bx����  �          @�=q@�=q�z�@*=qA��C�&f@�=q����@N{A�p�C�f                                    Bx���b  �          @�=q@�33���@#33A�  C��@�33��ff@H��A��C���                                    Bx���  �          @�G�@����p�@#33A���C�xR@��ÿ�\)@J=qA��
C�/\                                    Bx���  �          @ٙ�@��\��H@!G�A�ffC���@��\�˅@G�A�Q�C�]q                                    Bx���T  �          @ٙ�@�z��=q@
=A���C��H@�z�У�@=p�AΣ�C�J=                                    Bx���  �          @��@���Q�@��A�33C��@���˅@>�RA�{C�u�                                    Bx���  �          @�G�@�(�� ��@�A�
=C�o\@�(���  @:=qA���C���                                    Bx��"F  �          @ٙ�@�=q�+�@G�A��
C���@�=q��z�@<(�A���C��                                    Bx��0�  �          @�(�@�z��+�@�A�=qC��@�z��33@@  AθRC�                                    Bx��?�  �          @�(�@�Q���R@\)A�ffC�Ǯ@�Q�޸R@7
=Aģ�C��                                    Bx��N8  �          @��H@�(��#33@�A�{C�C�@�(���\@@  A�{C��q                                    Bx��\�  �          @�=q@�\)�\)@��A�=qC��3@�\)���
@0��A���C���                                    Bx��k�  �          @�  @�33��33@)��A���C���@�33����@E�A�\)C��                                    Bx��z*  �          @�\)@���
=@'
=A��C��@�녿��
@FffA�(�C��=                                    Bx����  �          @�Q�@�p�����@,(�A��C�y�@�p��z�H@EA��C�K�                                    Bx���v  �          @أ�@����
=@#�
A�G�C�.@���aG�@;�A���C���                                    Bx���  �          @���@��Ϳ��@   A��C��)@��Ϳ��@1G�A��HC�l�                                    Bx����  �          @أ�@�Q쿼(�@*=qA��C�#�@�Q�(��@=p�A���C�޸                                    Bx���h  �          @�G�@�����p�@#33A�=qC��@����p��@;�A�z�C��3                                    Bx���  �          @��@�  ��(�@5A�G�C�!H@�  �(�@HQ�AۮC�{                                    Bx���  T          @�(�@�녿   @~{B�RC�g�@��?!G�@}p�B  @�                                    Bx���Z  �          @�z�@��@  @s33B�C�9�@�>�{@w
=Bz�@Y��                                    Bx���   �          @���@���n{@_\)A�  C�|)@��=��
@g�A�p�?G�                                    Bx���  �          @��
@�G��Y��@b�\A��C��
@�G�>.{@h��B �?ٙ�                                    Bx��L  �          @�(�@�{��z�@dz�A���C��@�{��Q�@tz�BffC�.                                    Bx��)�  �          @�z�@�=q��Q�@uB��C���@�=q���
@���B�C���                                    Bx��8�  �          @�(�@������@^�RA�C�l�@����\)@j=qB �HC�Q�                                    Bx��G>  �          @�p�@��H�E�@g�A�=qC�AH@��H>��@l(�B �@!�                                    Bx��U�  �          @���@����@W�A�p�C��{@���Q�@a�A�C���                                    Bx��d�  �          @���@����H@Tz�A�C�8R@����@aG�A�p�C���                                    Bx��s0  �          @���@�
=���@L(�A���C��3@�
=�\@[�A��C�,�                                    Bx����  �          @�z�@�=q��\)@;�A�p�C��f@�=q��@L(�A���C��3                                    Bx���|  �          @�z�@�G���@Dz�A��
C���@�G���z�@QG�A��C��                                     Bx���"  �          @�z�@�Q쿐��@J�HA�\)C���@�Q�aG�@W
=A�C��3                                    Bx����  �          @�(�@\��\)@8��A�
=C���@\��@I��Aڏ\C��                                     Bx���n  �          @�z�@����aG�@H��A�p�C�� @���<#�
@P��A�z�=u                                    Bx���  �          @�ff@�  ���\@333A��HC�Z�@�  �k�@>{A�p�C��
                                    Bx��ٺ  �          @�@�ff��\)@'�A�(�C��=@�ff�Y��@>{A���C�3                                    Bx���`  �          @�p�@�ff��@33A�  C��@�ff��33@%A��C��                                    Bx���  �          @�{@�\)�{?��Ao�
C�.@�\)��33@��A�\)C��f                                    Bx���  �          @�\)@�  �!�?��Am��C�  @�  ����@��A���C�Q�                                    Bx��R  �          @�ff@�G��\)?�z�A~�HC�.@�G���33@(�A�Q�C���                                    Bx��"�  �          @�ff@�G���?���Ar{C�ٚ@�G���G�@�A��C�0�                                    Bx��1�  �          @�
=@ʏ\�p�?�ffAN=qC�ff@ʏ\����@	��A��C�l�                                    Bx��@D  �          @�Q�@�z��\)?�p�AB�HC�XR@�z�� ��@A��\C�E                                    Bx��N�  �          @�\)@ə��%�?\AI�C�� @ə���@	��A�\)C���                                    Bx��]�  �          @�\)@�33���?��AK�
C�s3@�33���H@Q�A�=qC�o\                                    Bx��l6  �          @���@���]p�?#�
@�\)C�#�@���J�H?��
AI��C�1�                                    Bx��z�  �          @�G�@��O\)?(�@���C�,�@��=p�?�Q�A>=qC�0�                                    Bx����  �          @��H@�(��^{?O\)@�G�C�>�@�(��H��?�Q�A\z�C�t{                                    Bx���(  T          @�=q@���Z�H?(��@��
C�}q@���G�?��
AHQ�C���                                    Bx����  �          @�=q@�����  ?�@�z�C���@����n{?���AQG�C��                                     Bx���t  �          @���@ʏ\�333?�G�A�C�
@ʏ\��?��HAaC�|)                                    Bx���  �          @�G�@�  ��?^�R@�\C�~�@�  �\?��A+\)C��                                    Bx����  �          @�  @�G�<�?�p�ADz�>k�@�G�>��?�A<��@\��                                    Bx���f  �          @�\@�G����Ϳ
=q���C��@�G���z�?�R@��C���                                    Bx���  �          @���@�����33�k���z�C���@�����>���@Q�C���                                    Bx����  �          @��@�
=���׿n{��RC�7
@�
=���>��@�C���                                    Bx��X  �          @�p�@��H��p������\C���@��H����>#�
?��\C��q                                    Bx���  �          @��@�p����\�������C��@�p���{>u?�C��q                                    Bx��*�  �          @��@�����\�p����\C��)@����p�>��@�
C��
                                    Bx��9J  �          @���@�������c�
��C�@������>�{@0  C��\                                    Bx��G�  �          @�p�@��R��33�0����Q�C��@��R���?�@���C�y�                                    Bx��V�  �          @�@��
��{�0����G�C�s3@��
���R?�@�z�C�b�                                    Bx��e<  �          @�(�@�����\)�k���C��R@�����=q>8Q�?�
=C�H�                                    Bx��s�  �          @���@�ff�w��J=q��(�C�}q@�ff�|(�>W
=?�z�C�<)                                    Bx����  �          @�(�@�  ��33�c�
����C�AH@�  ��{>.{?��C��{                                    Bx���.  �          @��@����{����Q�C�g�@����?(��@�=qC�t{                                    Bx����  �          @�z�@�(���p���z��C��@�(����H?aG�@�(�C��                                     Bx���z  �          @�z�@�����R���s33C�� @����33?�  A ��C���                                    Bx���   �          @��
@���������U�C��@�����?B�\@ÅC���                                    Bx����  �          @��
@����������s�
C��q@������H?.{@���C��3                                    Bx���l  T          @�33@��������  �   C��@������?n{@�C�Ff                                    Bx���  �          @�@�����H�Ǯ�H��C��
@����G�?@  @�33C���                                    Bx����  �          @�@������R��\)�{C���@�����z�?Tz�@ָRC��                                     Bx��^  �          @�33@����  �^�R�ᙚC���@����33>\)?���C�g�                                    Bx��  T          @�=q@��a녿�Q��<Q�C�� @��qG������\C��                                    Bx��#�  T          @�  @��\�H���Q����\C�ٚ@��\�c�
��ff�,(�C�G�                                    Bx��2P  �          @�z�@�{��ff�s33��C��@�{���S�
��\C��R                                    Bx��@�  �          @�33@�
=�AG��333���HC��=@�
=�g
=��p����\C�J=                                    Bx��O�  �          @ڏ\@��\�G
=�;���33C��q@��\�n�R�����C��                                    Bx��^B  �          @�=q@�=q�S�
�HQ�����C�q�@�=q�}p��\)���C��q                                    Bx��l�  �          @�G�@��
�HQ��J�H���HC�N@��
�s33����z�C���                                    Bx��{�  �          @أ�@��>�R�L(�����C�
=@��j=q�Q�����C�aH                                    Bx���4  �          @���@�G��=p��AG���  C�ff@�G��e��R���C���                                    Bx����  �          @���@�Q��Mp��P  ����C��@�Q��x��������C��                                    Bx����  �          @��H@�(��P  �*�H��Q�C�j=@�(��s33�����w�
C�e                                    Bx���&  �          @��H@�
=�k�>�=q@�\C��)@�
=�`��?��AC�7
                                    Bx����  �          @׮@�ff�{�?z�@�z�C��@�ff�k�?��RAK�C��q                                    Bx���r  �          @׮@�{�~{>�p�@J�HC��3@�{�qG�?��A1G�C��                                    Bx���  �          @�Q�@���z=q>�G�@qG�C�C�@���l��?���A8z�C�                                    Bx���  �          @���@�  �|��>�=q@  C�&f@�  �q�?�
=A ��C���                                    Bx���d  �          @�  @�������>8Q�?��
C���@����y��?�\)A��C�3                                    Bx��
  T          @׮@�p������k��   C�  @�p����?E�@ҏ\C�U�                                    Bx���  �          @�Q�@�{��녿!G���z�C���@�{���\>�G�@n�RC��
                                    Bx��+V  �          @�=q@��R��33�B�\���
C���@��R����>���@0��C�j=                                    Bx��9�  �          @ٙ�@����녿��
�.=qC�h�@������L�Ϳ�C��                                     Bx��H�  �          @���@�  ��\)�c�
��G�C��@�  ��=q>#�
?��C��                                    Bx��WH  �          @�G�@��R�������RC��@��R��z��G��s33C�u�                                    Bx��e�  �          @ٙ�@���  ���
=C�Ǯ@����ͽ�G��p��C�Q�                                    Bx��t�  �          @�G�@�����H����\C�K�@�������Q�J=qC�ٚ                                    Bx���:  �          @�33@��R��{�����W�C�{@��R��{������\C�P�                                    Bx����  �          @�33@����33�J=q��(�C���@�����>�=q@G�C�z�                                    Bx����  �          @��
@�p����Ϳ���� Q�C�H�@�p��������z�HC��{                                    Bx���,  �          @ٙ�@��������p��&�\C��@�����=q������
C�q�                                    Bx����  �          @�=q@��������Q��C
=C��@�����녾�p��HQ�C�z�                                    Bx���x  �          @��@�����  ��\)�7
=C�%@�����{�����0  C���                                    Bx���  �          @���@��H������H�!C�C�@��H���;8Q��G�C�Ǯ                                    Bx����  �          @�z�@����  ���\�)�C�'�@����p��u�   C��H                                    Bx���j  �          @�(�@��\��\)����� z�C�K�@��\��z�8Q��  C�Ф                                    Bx��  �          @��H@��\��(���=q�3�
C��q@��\��녾�{�5�C�f                                    Bx���  �          @ڏ\@��H��(���=q�4  C�#�@��H��녾�z����C���                                    Bx��$\  �          @��
@������ÿ���-p�C�y�@�����{�aG�����C���                                    Bx��3  �          @�(�@�����H��(��#\)C�7
@������\)��33C�Ǯ                                    Bx��A�  �          @�33@�  ��
=��z��>{C���@�  �����33�<(�C��                                    Bx��PN  �          @���@�Q���G���\)�7�
C�h�@�Q���
=�����!�C�޸                                    Bx��^�  �          @�z�@�(�����ff�.=qC��@�(���33����(�C���                                    Bx��m�  �          @�(�@�z���(���\)�8  C�L�@�z���녾�{�7�C��q                                    Bx��|@  �          @�33@����p���\)�8z�C��=@�������{�4z�C�\)                                    Bx����  �          @ۅ@����Q쿰���9��C�u�@����{�����2�\C��=                                    Bx����  �          @�33@�z���ff���
�qG�C�Ф@�z����R�0����G�C��                                    Bx���2  �          @��H@�33���ÿ˅�W�
C�q�@�33��  �   ����C��{                                    Bx����  T          @�(�@����\)��z��=��C�~�@�������\)�z�C�f                                    Bx���~  �          @�{@�Q���p���(��g
=C��q@�Q���������\)C��                                    Bx���$  
�          @�@����33�{���C�%@�����R��z��C�"�                                    Bx����  �          @�ff@�����=q�z���  C���@�����{���H� ��C��f                                    Bx���p  �          @�p�@�  ��{�z����\C�^�@�  ��녿��R�%G�C�XR                                    Bx��   �          @�(�@�������\���C�{@������  �'33C�f                                    Bx���  �          @��@�����  ���C�Q�@���p����H� ��C�K�                                    Bx��b  �          @��@�\)���������z�C���@�\)��(����  C���                                    Bx��,  �          @ۅ@�����H�	����(�C�  @������{���C��                                    Bx��:�  �          @�(�@�Q�����������C��@�Q����R��z��=�C��                                    Bx��IT  �          @�z�@�������.{��(�C��\@����
=�ٙ��d(�C�G�                                    Bx��W�  �          @�(�@�  ���H��H����C���@�  ��\)����:ffC���                                    Bx��f�  �          @��@�p���G�����(�C�Y�@�p����Ϳ��\�)�C�P�                                    Bx��uF  T          @�z�@�����/\)����C��@����z��  �j�HC���                                    Bx����  �          @��
@���z��7��ř�C��@����
����~�HC��q                                    Bx����  �          @�33@�z����
�:=q���C��@�z������
=����C���                                    Bx���8  �          @ۅ@�����8�����C�
@����H��z����C��\                                    Bx����  �          @ۅ@��
��Q��.�R���C�n@��
���R�޸R�k
=C�*=                                    Bx����  �          @�z�@�����\)�����C�P�@�����G������\)C�s3                                    Bx���*  �          @���@������\��R��=qC��R@�����
=��p��F=qC���                                    Bx����  �          @�p�@�=q���
�3�
��  C��=@�=q���\��{�yG�C�/\                                    Bx���v  �          @�p�@�������1G�����C�]q@���������s
=C�                                    Bx���  �          @�p�@�������(������C���@������H��
=�`��C�n                                    Bx���  �          @�p�@�\)���\�'
=���C�0�@�\)��  ��
=�`(�C���                                    Bx��h  �          @޸R@�����
�(����Q�C�R@����G��ٙ��b�\C�ٚ                                    Bx��%  
�          @�{@����  �(Q�����C��@�������
=�`  C��                                    Bx��3�  T          @߮@����  �'����
C�p�@�������z��\(�C�B�                                    Bx��BZ  �          @�G�@�����H�(���ffC�q�@�����R��p��B{C�aH                                    Bx��Q   �          @�G�@�  ��\)�����\C��@�  ��G���33��C�33                                    Bx��_�  �          @ᙚ@����녿�ff�K
=C�P�@����Q�����
C�                                    Bx��nL  �          @�=q@�p����
��33�6�HC�0�@�p���G���
=�W�C��R                                    Bx��|�  T          @ᙚ@�33�������,��C��@�33���\�����-p�C�aH                                    Bx����  T          @��@����������R�D  C�P�@������������C��=                                    Bx���>  T          @�Q�@�=q��녿�33�YC��@�=q���׿0�����C�q�                                    Bx����  T          @�  @�����(���(��B{C��\@�����=q�   ���HC�0�                                    Bx����  �          @�  @�\)����Q��=��C�p�@�\)��33���q�C��R                                    Bx���0  T          @�Q�@�Q���p����H�@��C��=@�Q���33���H��G�C�                                    Bx����  T          @���@�G����ÿs33���C�^�@�G����<��
>.{C�#�                                    Bx���|  �          @�\)@�z���Q쿪=q�/�C��3@�z������33�:=qC���                                    Bx���"  �          @�\)@�p���ff��Q��?
=C�33@�p����
���}p�C���                                    Bx�� �  
�          @�
=@�=q��33���
�)C��@�=q�����33�6ffC��f                                    Bx��n  �          @�
=@�ff��\)��ff�+�C���@�ff��(��Ǯ�J�HC�>�                                    Bx��  �          @߮@����ff��\)�\)C��@����=q�aG���C�                                    Bx��,�  �          @��@�G����H�
=���HC��H@�G����>��
@'
=C���                                    Bx��;`  �          @��@�p���\)���
�'�C�޸@�p���ff?(�@��RC��                                    Bx��J  �          @�Q�@�����ff�
=���C��@�����\)>���@,(�C��R                                    Bx��X�  �          @�
=@����z�+�����C�)@����>k�?�C�                                      Bx��gR  T          @޸R@�=q����s33��z�C��@�=q����#�
��Q�C���                                    Bx��u�  �          @�
=@��H��33���H��
C���@��H��\)���R�!�C��H                                    Bx����  �          @߮@�{������\)��C�p�@�{��p��k���z�C�q                                    Bx���D  �          @�
=@�����\��p��#33C�)@�����R��{�5�C��)                                    Bx����  
�          @޸R@�z���Q쿜(�� ��C��@�z���zᾙ���{C��R                                    Bx����  �          @�\)@����  �������C�E@����33�#�
����C��q                                    Bx���6  �          @�{@��������ff��C��R@�����;\)��
=C�t{                                    Bx����  �          @���@����33�u� (�C�1�@����{�L�;ǮC��R                                    Bx��܂  �          @�@����  ����Q�C�<)@����33�����p�C���                                    Bx���(  �          @޸R@����\)���R�$(�C�k�@��������R�"�\C��                                    Bx����  �          @�Q�@����{��  �#�C��R@����=q�����*=qC���                                    Bx��t  �          @�Q�@�\)��{����)�C��@�\)���\�\�Dz�C���                                    Bx��  �          @ᙚ@�����{�����4��C�3@������H����s�
C���                                    Bx��%�  �          @��@�  ������
�'
=C��H@�  ���
��Q��=p�C��=                                    Bx��4f  �          @�=q@�(���{��=q���C�j=@�(������8Q쿺�HC�&f                                    Bx��C  �          @ᙚ@�����H�}p��G�C��@������G��h��C��f                                    Bx��Q�  �          @�=q@�G���G���z���C�U�@�G���������=qC�f                                    Bx��`X  �          @��@�{��G���{�3
=C�~�@�{��{�
=q��(�C��                                    Bx��n�  �          @�Q�@����׿�33�8Q�C���@����z���  C��                                    Bx��}�  �          @��@���������N{C��R@����@  ���C��                                    Bx���J  �          @߮@�(���������S
=C�g�@�(����J=q�θRC��                                     Bx����  T          @�\)@�(����R��33�[
=C���@�(�����Y�����C��{                                    Bx����  �          @�  @�p���
=��{�T(�C�� @�p�����O\)����C�
                                    Bx���<  �          @߮@��R�����
�JffC��
@��R����@  ���
C�U�                                    Bx����  �          @�\)@�ff��{��  �E�C��@�ff����5���
C�P�                                    Bx��Ո  �          @�  @��R��z��
=�]�C���@��R���H�fff��C�ff                                    Bx���.  �          @�
=@�\)���ÿ�{�w�C�Y�@�\)��Q쿎{�=qC��                                    Bx����  �          @߮@����녿����5C�+�@�����R�
=���\C��H                                    Bx��z  �          @�Q�@�Q�������;�C���@�Q���녿!G���(�C�(�                                    Bx��   �          @�33@�ff��Q��\�g�C��R@�ff��
=��  �p�C��                                    Bx���  �          @�(�@�z���z��z��W�C��@�z����\�^�R��Q�C���                                    Bx��-l  �          @�(�@����z��\�f{C��\@����33�z�H��{C�`                                     Bx��<  T          @��
@�(���녿��z{C�7
@�(���G���33�(�C���                                    Bx��J�  T          @�\@����녿��|z�C��@����G���z���\C�c�                                    Bx��Y^  �          @�G�@������H����k�C���@����������\�C�8R                                    Bx��h  �          @�G�@�{��
=��\)�T��C�/\@�{���ͿW
=��33C��{                                    Bx��v�  �          @ᙚ@�{��  ��ff�K�C�
@�{��p��E���G�C���                                    Bx���P  �          @�\@�ff���\����5��C��@�ff��
=�(���{C���                                    Bx����  �          @�\@�\)����� ����G�C��)@�\)��Q쿢�\�&�HC�4{                                    Bx����  �          @�(�@����{�a���C��{@������9�����
C�5�                                    Bx���B  �          @�{@��������N�R��C���@�����
=�%���HC��=                                    Bx����  �          @�R@��\��{�?\)��(�C��{@��\��=q���G�C���                                    Bx��Ύ  �          @�{@����p��6ff����C�y�@�������
�H��Q�C�n                                    Bx���4  �          @�z�@�G���33�&ff��33C���@�G���p���33�w33C��                                    Bx����  �          @��
@�������p���  C�O\@������R��G��d��C�z�                                    Bx����  T          @�z�@����(������HC���@����p���p��`Q�C��                                     Bx��	&  �          @��
@�=q������\)C�ff@�=q���R���X��C��q                                    Bx���  �          @�@�  ���R�Q���z�C�  @�  �����
=�Z�HC�Y�                                    Bx��&r  �          @�@����ff������C��@����\)���H�^=qC�W
                                    Bx��5  �          @�@�z����H�33����C�` @�z�����˅�O
=C��=                                    Bx��C�  �          @��
@��R���
�
=��C���@��R�����33�5��C���                                    Bx��Rd  �          @�(�@��R��z����(�C�z�@��R��(������2�HC���                                    Bx��a
  �          @��H@���������\���C�
@�����  ��{�0��C�w
                                    Bx��o�  �          @�\@��H�����
�G�C�<)@��H���H�^�R����C�Ǯ                                    Bx��~V  �          @��H@�=q��  ���9G�C�  @�=q��z�B�\���
C��R                                    Bx����  �          @�\@�����Ϳ�=q�N{C�b�@����녿n{���C��=                                    Bx����  �          @�=q@�
=��
=�޸R�c33C�>�@�
=�����{�z�C���                                    Bx���H  �          @�@�33��G�������C�.@�33��  ��{�0  C��f                                    Bx����  �          @�(�@�  ����33��  C�'�@�  ���H��p��?�
C�o\                                    Bx��ǔ  �          @�(�@�
=��
=��Q��|��C�:�@�
=����=q�+�C��)                                    Bx���:  �          @�@��
���
��p��a�C��q@��
��������\)C�o\                                    Bx����  �          @�(�@��R�������HQ�C�<)@��R���׿s33��{C�                                    Bx���  �          @��@�����(������.=qC�Z�@������׿B�\��(�C��3                                    Bx��,  �          @�(�@���G��c�
����C�\@�������
�#�
C��{                                    Bx���  
�          @�@�
=����+���(�C�N@�
=��G���G��aG�C�'�                                    Bx��x  T          @ᙚ@��\��=q�+����C��\@��\���
��G��fffC��=                                    Bx��.  �          @�Q�@�z���{�W
=��p�C�Ф@�z���Q쾊=q�(�C��)                                    Bx��<�  �          @�\)@��H�������\)C���@��H��  ���H��  C�z�                                    Bx��Kj  �          @�
=@��\��(���33�z�C���@��\��\)�
=��33C���                                    Bx��Z  �          @�
=@�{��\)����+�C�q@�{��33�8Q���ffC��H                                    Bx��h�  �          @�ff@�����ff��  �F�HC��H@������H�fff��{C�9�                                    Bx��w\  �          @�{@�G���z�p�����C�'�@�G���
=����Y��C��f                                    Bx���  �          @޸R@�
=�����
=q��\)C�s3@�
=���\�#�
�L��C�Y�                                    Bx����  �          @�\)@�������L���ҏ\C��@���������=q��RC��3                                    Bx���N  �          @�Q�@������ÿ#�
��  C���@������\��G��n{C��R                                    Bx����  �          @��@��H��\)�0����z�C�  @��H���þ.{��33C��R                                    Bx����  �          @�Q�@�G���  �����C�@ @�G���G��#�
��p�C�#�                                    Bx���@  �          @�
=@�����þ��
�*=qC��@����G�>#�
?�=qC��                                    Bx����  T          @޸R@��H����k���33C���@��H��\)�Ǯ�N{C��=                                    Bx���  �          @�G�@�ff��  ��  �#�C�f@�ff����B�\��  C���                                    Bx���2  �          @޸R@�����\����Y��C��=@����33=�\)?z�C���                                    Bx��	�  �          @�ff@�{��p����R�%G�C���@�{���ÿ@  ��ffC�p�                                    Bx��~  �          @�
=@�Q���  �#�
����C�,�@�Q����>���@ ��C�4{                                    Bx��'$  T          @�  @����;�����C��H@�����>8Q�?�  C�|)                                    Bx��5�  �          @�Q�@�  ��33��\)�\)C���@�  ���\>���@QG�C���                                    Bx��Dp  �          @ᙚ@��H��G��L�;�p�C�:�@��H����>��@W�C�J=                                    Bx��S  �          @ᙚ@�
=��p�>�?���C��f@�
=��(�?��@�p�C��f                                    Bx��a�  �          @�\@��H��  ?B�\@ƸRC�]q@��H����?��HA��C���                                    Bx��pb  �          @�=q@����(�?u@��HC��@����  ?��A5��C�Z�                                    Bx��  �          @ᙚ@�������?&ff@���C��@������R?��A
=C�j=                                    Bx����  �          @�G�@����  ?.{@��C�L�@�����?�\)A=qC���                                    Bx���T  �          @��H@������
?=p�@���C��\@�������?�Q�A�\C�!H                                    Bx����  �          @�\@�����Q�?G�@ʏ\C��@������?��RA ��C�`                                     Bx����  �          @ᙚ@�����=q?!G�@�z�C��q@������?���A33C�E                                    Bx���F  �          @��@�ff��\)>�?�=qC�L�@�ff��{?z�@�\)C�j=                                    Bx����  �          @�\)@�z����;����S33C�aH@�z���p�=#�
>���C�S3                                    Bx���  �          @޸R@�\)����>\)?�z�C��q@�\)��  ?\)@���C��                                    Bx���8  �          @�\@����(��#�
����C���@�����>Ǯ@K�C���                                    Bx���  �          @��H@��\���>aG�?�C��@��\���?&ff@��C�'�                                    Bx���  �          @�=q@������>.{?�\)C�S3@�����?z�@�\)C�p�                                    Bx�� *  �          @�=q@����s�
>�=q@
�HC��\@����qG�?!G�@���C���                                    Bx��.�  �          @߮@���������G�C�Q�@�����p�>���@��C�XR                                    Bx��=v  �          @���@�=q��33���
��C�!H@�=q���\>�G�@g�C�0�                                    Bx��L  �          @�=q@��
������
�!G�C�9�@��
��33>\@E�C�C�                                    Bx��Z�  �          @ᙚ@�\)���R��p��B�\C��\@�\)��
==�Q�?:�HC��                                    Bx��ih  �          @���@�z���Q�(���Q�C�(�@�z���G������(�C�\                                    Bx��x  �          @���@���zῆff���C�*=@����R�����C���                                    Bx����  �          @߮@�z��������H��
C�Ф@�z���z�8Q����RC���                                    Bx���Z  �          @�\)@�
=��  �������C�5�@�
=��=q�
=���C��                                    Bx���   �          @�\)@���G������C�w
@������R���HC�@                                     Bx����  �          @߮@������s33��33C���@����\)�
=q���C���                                    Bx���L  �          @���@�Q���녿�G���
C��@�Q���(��z���\)C��H                                    Bx����  �          @��@�����\)�����(�C�l�@�����녿.{��=qC�0�                                    Bx��ޘ  �          @߮@�(���(���ff�
�RC�|)@�(���ff�!G�����C�Ff                                    Bx���>  �          @��@�p����z���  C���@�p����R�8Q쿸Q�C��\                                    Bx����  �          @�=q@�z������u��
=C���@�z�����>��?�(�C��f                                    Bx��
�  �          @��@�=q���>��
@%�C�&f@�=q��Q�?.{@���C�J=                                    Bx��0  �          @�=q@��R���R>�\)@\)C�^�@��R��?&ff@���C�}q                                    Bx��'�  �          @�G�@�p�����?G�@���C�y�@�p���=q?�33A=qC���                                    Bx��6|  �          @��@����=q?�33A33C���@�����R?�  AF{C��                                    Bx��E"  �          @�  @�
=��
=>��@xQ�C�*=@�
=��p�?O\)@׮C�W
                                    Bx��S�  �          @�\)@����
���
��RC��
@����>�\)@�\C��q                                    Bx��bn  �          @�
=@�{��  =�\)?��C�s3@�{�~�R>Ǯ@O\)C���                                    Bx��q  �          @�  @����i��?�=qA/�C�޸@����a�?�{AV{C�H�                                    Bx���  �          @�\)@����j�H?���A�HC��R@����c�
?��RAEG�C�9�                                    Bx���`  �          @޸R@������?�  AG�C��@���{�?���A/
=C�`                                     Bx���  �          @�ff@�\)��=q>B�\?��C�o\@�\)��G�?
=q@��RC��f                                    Bx����  �          @�ff@����z�>L��?�{C�Y�@�����?�@���C�p�                                    Bx���R  �          @�@����{�?��RA%�C�K�@����u�?��AN{C��=                                    Bx����  �          @�ff@��n{?�p�AEG�C�g�@��e?�G�Aj�RC��
                                    Bx��מ  �          @߮@�p��S33?�A]�C�o\@�p��J=q?�z�A}�C��\                                    Bx���D  �          @�
=@���@  ?�
=A��\C���@���5@	��A��C�K�                                    Bx����  �          @�
=@����A�?ٙ�Ab�\C���@����9��?�A�(�C�&f                                    Bx���  �          @޸R@���Y��?�ffANffC���@���QG�?��Ao\)C�h�                                    Bx��6  �          @�{@Å�7�?�=qAR�\C�k�@Å�0  ?��
An=qC���                                    Bx�� �  �          @��@Å�<(�?�p�A$z�C�#�@Å�6ff?�Q�A@z�C��H                                    Bx��/�  �          @ۅ@�z��R�\?���A"{C�e@�z��L��?�Q�AAp�C���                                    Bx��>(  �          @�(�@�33�u�?���A
=C���@�33�o\)?��A;
=C�"�                                    Bx��L�  �          @�(�@�p���33?h��@�(�C�w
@�p�����?�Q�A Q�C���                                    Bx��[t  �          @�(�@�  �hQ�?�A��C�H�@�  �_\)@
=qA�
=C��                                    Bx��j  �          @�(�@�ff�]p�@33A�
=C��=@�ff�S�
@�A��C�Z�                                    Bx��x�  �          @���@���j=q?���A"{C��\@���dz�?���AC\)C��                                    Bx���f  �          @�@��H��33=���?Tz�C�� @��H���\>�p�@FffC��                                    Bx���  �          @߮@�p����\>���@�RC�'�@�p�����?�@�ffC�AH                                    Bx����  �          @�\)@����z�>L��?�33C�Ǯ@�����
>��@x��C���                                    Bx���X  �          @�ff@�
=��G������G�C���@�
=��G�>�?�ffC���                                    Bx����  
�          @�ff@�
=���\����G�C�\)@�
=���\>.{?�
=C�]q                                    Bx��Ф  �          @���@�\)��>��
@(Q�C�Q�@�\)����?z�@��HC�j=                                    Bx���J  �          @ۅ@�z��B�\?�G�AmC�U�@�z��;�?���A�  C���                                    Bx����  �          @�z�@�=q�S�
?���AVffC�0�@�=q�L��?�ffArffC���                                    Bx����  �          @�p�@�����?O\)@�\)C�� @����Q�?��AG�C���                                    Bx��<  �          @�\)@�  ��������(�C�u�@�  ���<��
>�C�o\                                    Bx���  �          @�
=@�ff��(����ͿW
=C�<)@�ff��(�>W
=?�(�C�>�                                    Bx��(�  �          @޸R@�=q��Q�\)��
=C��@�=q��Q�>#�
?�ffC���                                    Bx��7.  T          @�@������>8Q�?��RC���@����Q�>�ff@n{C���                                    Bx��E�  �          @޸R@�33��ff>�z�@ffC��@�33��p�?��@�  C�33                                    Bx��Tz  �          @޸R@�{���
=#�
>�33C�*=@�{���>��R@#�
C�33                                    Bx��c   �          @�ff@�  ����<�>aG�C�
@�  ��G�>���@   C�q                                    Bx��q�  T          @�ff@�
=���H<��
>8Q�C�XR@�
=���\>�\)@�C�^�                                    Bx���l  �          @޸R@�ff���\=�Q�?@  C���@�ff���>���@0��C��3                                    Bx���  �          @�
=@�{����=L��>�
=C�c�@�{��Q�>�=q@��C�k�                                    Bx����  �          @�\)@�{���>8Q�?�Q�C���@�{��33>�
=@Z�HC���                                    Bx���^  �          @�p�@����Q�>�Q�@<(�C���@����\)?��@�\)C��)                                    Bx���  �          @޸R@�{����?Q�@���C�y�@�{��
=?��A��C���                                    Bx��ɪ  �          @�\)@��
��p�?@  @ƸRC��3@��
���
?�  A(�C��R                                    Bx���P  �          @�\)@�����H>���@�RC�h�@�����?
=q@��RC�z�                                    Bx����  �          @�G�@��R���R��{�3�
C��@��R��
=���ͿL��C��                                    Bx����  �          @�=q@��R��Q쾞�R� ��C��=@��R���׽u��\C�                                    Bx��B  �          @�=q@��
���
��
=�[�C�7
@��
��(��.{��z�C�+�                                    Bx���  �          @�
=@��\������
�#�
C��@��\���>W
=?ٙ�C��{                                    Bx��!�  T          @޸R@�����{>��
@(Q�C�� @������?�@��HC���                                    Bx��04  �          @�\)@������>�{@1�C�<)@�����
=?��@��C�N                                    Bx��>�  �          @�@��
����    =#�
C�� @��
����>k�?�z�C��                                    Bx��M�  �          @�ff@�����\)�����HC�p�@�����  �������C�aH                                    Bx��\&  �          @�{@����p��z���Q�C�XR@����{��{�4z�C�G�                                    Bx��j�  �          @�ff@�Q���  �L�����
C��@�Q���G�����p�C��                                    Bx��yr  �          @�@�{�����G��'�C�Z�@�{���R���\�33C�5�                                    Bx���  �          @�@����ff��G��((�C�&f@����  ���\�  C�H                                    Bx����  �          @�p�@�(���
=��33���C�H@�(����׿h�����C��H                                    Bx���d  �          @�{@�����
����	G�C��
@����p��O\)��p�C���                                    Bx���
  �          @�
=@�z����ͿO\)����C�|)@�z����
=���C�e                                    Bx��°  �          @޸R@�����
=�O\)��
=C��)@�����  ������C��                                    Bx���V  T          @�ff@�Q���Q�z����C�<)@�Q����þ\�G
=C�,�                                    Bx����  �          @�{@�33����(����C�Ǯ@�33������Z=qC���                                    Bx���  �          @�p�@�p���=q����w
=C�>�@�p����H��=q�G�C�33                                    Bx���H  �          @�{@�(����
�&ff��(�C�  @�(���z���r�\C��                                    Bx���  �          @߮@�p���  ���\�
=C���@�p���G��Y����\)C���                                    Bx���  �          @߮@�ff���
��33�Z�RC���@�ff����(��B�HC��f                                    Bx��):  �          @�\)@��
��p���
=�_
=C���@��
��\)��  �G
=C�ff                                    Bx��7�  �          @�  @�Q���녿��H�b=qC��{@�Q���(����
�I�C���                                    Bx��F�  �          @�  @�G���녿�
=�^�RC�k�@�G����
���R�EG�C�<)                                    Bx��U,  �          @���@�(����׿���W�C�S3@�(����\��Q��=C�'�                                    Bx��c�  �          @���@������
��33�Y�C��)@��������H�?�
C���                                    Bx��rx  �          @��@��������
�jffC��
@����녿����RffC��f                                    Bx���  �          @�  @��������G���C�1�@�����33�����HC���                                    Bx����  �          @�\)@�\)��������@  C�y�@�\)���R���
�)G�C�Q�                                    Bx���j  �          @�  @����p���\�k33C��@����\)�˅�R{C��                                    Bx���  �          @���@�33��=q��
=�<z�C���@�33������R�#\)C���                                    Bx����  �          @��@�(���z῜(���
C��@�(�������(�C��f                                    Bx���\  �          @�\)@�G����ÿ�33�[33C�H@�G����H��p��C�
C���                                    Bx���  �          @�  @�����\)����{
=C�33@�����G���p��d(�C��                                    Bx���  �          @���@������H�����1G�C��@�����(���Q���
C��                                    Bx���N  �          @���@�������\�FffC�q�@������
�k���\)C�j=                                    Bx���  �          @�  @�ff��z�
=q���C�"�@�ff����Ǯ�K�C��                                    Bx���  �          @�G�@��H����s33����C���@��H��Q�O\)��33C��                                    Bx��"@  �          @�  @�����Q�=p��\C�Ǯ@������ÿ����p�C��
                                    Bx��0�  �          @�\)@�����H��{��C�� @�����
�xQ��   C�e                                    Bx��?�  �          @߮@��\�~�R�����.=qC�AH@��\���׿�����C�!H                                    Bx��N2  �          @�
=@�����=q����-C�k�@��������
=�Q�C�L�                                    Bx��\�  T          @޸R@����(���=q�R{C��R@����p������@��C��3                                    Bx��k~  �          @޸R@�����Ϳ�z��\  C�H�@����ff�\�IC�#�                                    Bx��z$  �          @�  @��������\)�U��C��
@���������R�D  C��{                                    Bx����  �          @��@�p����׿���JffC��@�p���녿�33�8z�C�Ф                                    Bx���p  �          @�Q�@�=q��(����:�HC�T{@�=q��p����
�(��C�8R                                    Bx���  �          @��@������R��{�2=qC���@�����  ��(�� Q�C��R                                    Bx����  �          @�Q�@�\)��녿�=q�p�C��f@�\)���H�p����
=C�s3                                    Bx���b  �          @�Q�@�  ��Q쿝p��!p�C���@�  ��G������  C���                                    Bx���  �          @�G�@�������  �Dz�C�"�@������R��\)�3�C�f                                    Bx���  �          @��@�G���ff��{�3
=C��@�G���\)���R�"=qC��\                                    Bx���T  �          @�Q�@������Ϳ�z��9p�C�4{@��������
�)�C�)                                    Bx����  �          @�
=@�{�������+
=C��R@�{���ÿ���RC��H                                    Bx���  �          @޸R@�p���G���\)�  C�ff@�p���=q��  ��
C�S3                                    Bx��F  �          @߮@�Q����H��z��[33C�@�Q���(�����J�HC���                                    Bx��)�  �          @�\)@����녿ٙ��b{C��@����33�˅�R=qC��3                                    Bx��8�  �          @�
=@�(���(����
�l(�C�>�@�(���p���z��\(�C�"�                                    Bx��G8  �          @߮@������R����mC��@�����  ��
=�]�C��=                                    Bx��U�  �          @�  @�  ��������u��C��@�  ���ÿ޸R�e�C�j=                                    Bx��d�  �          @߮@�  �����=q�rffC��@�  ���ÿ��H�c
=C�k�                                    Bx��s*  �          @�Q�@�\)��\)��(�����C�}q@�\)���׿�{�v�\C�aH                                    Bx����  �          @ᙚ@����Q�����s
=C��\@����G��޸R�dQ�C��
                                    Bx���v  �          @�G�@����G�����x��C�^�@�����\���
�jffC�E                                    Bx���  �          @ᙚ@�  ���ÿ�����(�C�e@�  ��=q���r{C�L�                                    Bx����  �          @�G�@�=q���R���
�jffC�]q@�=q������\  C�G�                                    Bx���h  �          @ᙚ@�(����Ϳ���w33C���@�(���{���
�iG�C���                                    Bx���  �          @��@������
��z��{
=C��
@���������m��C��                                     Bx��ٴ  �          @ᙚ@\)�����33�zffC��@\)���׿�ff�l��C���                                    Bx���Z  �          @�=q@����{�ٙ��_\)C���@����
=��{�RffC���                                    Bx���   �          @�\@����������V�\C�&f@�����{��ff�I�C�{                                    Bx���  �          @��@���������[�C�Ff@�����z��=q�O\)C�33                                    Bx��L  �          @��@���������H�_�C�H�@�����z��\)�S�C�7
                                    Bx��"�  �          @�=q@�(���\)�����s
=C���@�(���Q��\�g�C��                                    Bx��1�  �          @�=q@���\)��p��b{C�"�@���Q����V�HC��                                    Bx��@>  �          @�\@����
=��33�W\)C�U�@����  �����Lz�C�E                                    Bx��N�  T          @�\@�\)��
=���Y�C�P�@�\)��  �˅�O33C�@                                     Bx��]�  T          @��
@��
���
��
=�YC��\@��
���Ϳ����O
=C�~�                                    Bx��l0  T          @�(�@�
=��녿�\)�RffC��@�
=���\����H(�C���                                    Bx��z�  �          @���@�
=��  ��G��d(�C���@�
=���׿�
=�YC��{                                    Bx���|  �          @��@������H����uG�C��{@���������k
=C��                                    Bx���"  �          @��@��
���ÿ��y�C�]q@��
���������o33C�L�                                    Bx����  �          @�(�@h��������Q��}��C��@h����=q��\)�s\)C��                                    Bx���n  �          @�z�@�p���녿\�EC�o\@�p����\�����<Q�C�c�                                    Bx���  �          @�z�@����=q���\�$Q�C��@�����H���H�(�C��R                                    Bx��Һ  �          @�@�=q���H�&ff����C��
@�=q��33������HC���                                    Bx���`  �          @�Q�@����z=q���k�C��@����z=q<��
>��C��                                    Bx���  �          @���@����hQ�>�?���C�u�@����hQ�>.{?�z�C�w
                                    Bx����  �          @��@�p��Z�H<�>�=qC�y�@�p��Z�H=�\)?
=C�z�                                    Bx��R  �          @��@��H�Fff���
���C���@��H�Fff<��
=�C���                                    Bx���  �          @ᙚ@��
�@�׾W
=��Q�C�Y�@��
�@�׾8Q쿷
=C�XR                                    Bx��*�  �          @ᙚ@�{�6ff�8Q쿸Q�C��@�{�6ff�������C��                                    Bx��9D  T          @ᙚ@���;��B�\����C��
@���;��#�
��=qC���                                    Bx��G�  �          @��@�p��Z�H��Q��:=qC���@�p��Z�H�����(��C��                                     Bx��V�  �          @��@��W
=�
=q���
C���@��W
=��\���C���                                    Bx��e6  �          @��H@�\)�333����(�C�U�@�\)�333�����p�C�P�                                    Bx��s�  �          @��@ƸR�S33���q�C��@ƸR�S33��G��b�\C���                                    Bx����  �          @��H@�
=�X�ý#�
��Q�C���@�
=�X�ü��
��C���                                    Bx���(  �          @��
@��H�N{����z�C���@��H�N{�k���{C��f                                    Bx����  �          @�z�@��
�L�;�p��=p�C���@��
�L�;�{�0��C��                                    Bx���t  �          @�z�@�ff�AG�>B�\?��C�o\@�ff�AG�>W
=?�(�C�p�                                    Bx���  �          @�=q@�G��L��>�33@5C��f@�G��L(�>�p�@A�C���                                    Bx����  �          @��H@���y��>.{?���C�L�@���y��>L��?���C�N                                    Bx���f  �          @�\@����z=q>8Q�?���C�5�@����z=q>W
=?�z�C�7
                                    Bx���  �          @�33@�p��x��>��R@"�\C�T{@�p��x��>�{@/\)C�U�                                    Bx����  �          @�p�@�  �s�
?=p�@�C��@�  �s33?B�\@��
C���                                    Bx��X  �          @�@���tz�>��@o\)C��@���s�
>��H@z�HC��                                    Bx���  �          @��@���  �W
=���C��q@���  �8Q쿺�HC��q                                    Bx��#�  �          @��
@�����p��E��ǮC�>�@������@  ����C�<)                                    Bx��2J  �          @��@�z����\��(��G�C�c�@�z����H�����{C�^�                                    Bx��@�  T          @��@�����
��=q��C�� @�����
����z�C��)                                    Bx��O�  �          @��@��R��ff��Q��9�C���@��R��ff���733C���                                    Bx��^<  �          @��@�{��
=���6�HC�޸@�{��
=��33�4(�C���                                    Bx��l�  �          @�p�@�p���ff�����N�\C��q@�p���ff��=q�L  C��R                                    Bx��{�  �          @�p�@�G����Ϳ����1G�C���@�G������{�.�HC��                                     Bx���.  �          @�R@����33��=q�*ffC�]q@���������(  C�Y�                                    Bx����  �          @�@�=q�����(��=�C��@�=q���
�����;
=C��                                    Bx���z  �          @�R@�����{��33�3�C�,�@�����ff����1��C�(�                                    Bx���   �          @�\)@������R���H��\C���@������R�������C��)                                    Bx����  �          @�  @�����\���\�!p�C�J=@�����H��G��   C�G�                                    Bx���l  �          @��@�G���  �����'33C�u�@�G���Q쿧��%C�s3                                    Bx���  �          @�Q�@��H��\)����&�\C�>�@��H��\)��ff�%�C�<)                                    Bx���  �          @�  @�ff��=q��G��@  C���@�ff��=q��  �>�RC��
                                    Bx���^  �          @�R@����R��  �`��C�h�@���
=�޸R�_�C�ff                                    Bx��  �          @�\)@�����(����
�d��C���@�����z���
�c�C��H                                    Bx���  �          @�  @�p���  ���
�c\)C�Ff@�p���  ��\�bffC�E                                    Bx��+P  �          @�\)@�����(���(��]�C��@�����z��(��\Q�C��                                     Bx��9�  �          @�@���Q��33�S�C���@����׿�33�S
=C���                                    Bx��H�  �          @�ff@�p���(���z��v�\C�+�@�p���(���z��u�C�*=                                    Bx��WB  T          @�  @�33���H�޸R�^�HC�AH@�33���H�޸R�^ffC�@                                     Bx��e�  �          @�  @��R��\)��Q��X  C���@��R��\)��Q��W�C��                                    Bx��t�  �          @�
=@�{��  �\)����C��{@�{��  �\)���\C��3                                    Bx���4  �          @�@�p���  �޸R�_33C��
@�p���  �޸R�_33C��
                                    Bx����  �          @�
=@�\)��=q��\��G�C���@�\)��=q��\��G�C���                                    Bx����  �          @�
=@�(�����G���(�C���@�(�����G���=qC���                                    Bx���&  �          @�  @�����R��  �>ffC�5�@�����R��  �>�RC�5�                                    Bx����  �          @�Q�@�������\��p�C���@�������\����C���                                    Bx���r  �          @�@�(���
=��
=�w\)C���@�(���
=��
=�w�C���                                    Bx���  �          @�
=@�Q���33�{��{C�9�@�Q���33��R��ffC�<)                                    Bx���  �          @�p�@��H���H������C���@��H���H�Q����C��q                                    Bx���d  �          @�p�@�z������z��w\)C��@�z���33���x  C�
                                    Bx��
  
�          @�{@�  ���
��z��T��C�W
@�  ���
��z��UC�XR                                    Bx���  �          @�R@����p�������C���@����p������C��                                     Bx��$V  �          @�
=@�����녿��R�?33C�k�@�����녿�  �@  C�n                                    Bx��2�  �          @�{@�\)��녿�(��]�C��@�\)��녿�p��^�HC��=                                    Bx��A�  �          @�R@����
=��z��v=qC��3@����
=���w\)C���                                    Bx��PH  �          @�{@�  ��\)� ������C�\)@�  ��\)�G���Q�C�^�                                    Bx��^�  �          @���@�{���ÿ�(���(�C���@�{���׿�p�����C��                                    Bx��m�  
�          @�@�\)���ÿ��H�}�C���@�\)���׿�p���C��f                                    Bx��|:  �          @�{@�z����\� ����z�C�8R@�z���=q�G���G�C�:�                                    Bx����  �          @�ff@�{����Q��Z{C�
@�{��p����H�[�C��                                    Bx����  �          @�ff@�(����ÿ����MG�C�\@�(����ÿ�{�O
=C�3                                    Bx���,  �          @�
=@�G���ff���R�>{C���@�G���{��  �?�
C�˅                                    Bx����  �          @�R@��\���Ϳ�p��=�C��@��\��z��  �?�C�                                    Bx���x  �          @�
=@�(��������%��C��R@�(���33����'\)C���                                    Bx���  �          @�Q�@����{�����C���@����{��
=�=qC��{                                    Bx����  �          @�  @�����=q���
�
=C�4{@�����=q��ff���C�7
                                    Bx���j  �          @�  @�������\(���G�C�]q@�������^�R����C�`                                     Bx��   T          @�  @��
��Q�z�H��Q�C�9�@��
��Q�}p���(�C�<)                                    Bx���  �          @���@�(���녿h����ffC�
@�(���녿n{��\C�R                                    Bx��\  �          @���@�z����R�����'\)C�l�@�z����R����)p�C�o\                                    Bx��,  �          @���@�G������=q�HQ�C��@�G�����˅�JffC��                                    Bx��:�  �          @���@��H�����   �\)C�W
@��H��G��G����HC�\)                                    Bx��IN  �          @�G�@�33�����G��_�C��@�33������
�a�C��                                    Bx��W�  �          @��@�ff��{��p��
=C�9�@�ff��{���R���C�=q                                    Bx��f�  �          @���@����������C�
C��q@�������Ǯ�FffC���                                    Bx��u@  �          @陚@�z����H�h����C��f@�z����\�n{���HC���                                    Bx����  T          @陚@����G�����\)C���@�����ÿ�z��{C��q                                    Bx����  �          @�=q@����z῝p���HC�G�@����zῠ  ���C�K�                                    Bx���2  �          @陚@��R��p���z��2ffC���@��R��p���
=�5�C���                                    Bx����  �          @��@��\���׿�G��>�\C���@��\���׿��
�AG�C���                                    Bx���~  �          @陚@��
��Q쿵�4  C���@��
��  ��Q��6�RC���                                    Bx���$  �          @���@�Q����ÿ����J�HC�q�@�Q����ÿ�\)�M��C�u�                                    Bx����  �          @���@�����������J�RC�T{@����G���\)�Mp�C�Y�                                    Bx���p  �          @�G�@�p���p���
=�5��C�33@�p���p����H�8Q�C�8R                                    Bx���  �          @��@�=q��{��ff�dQ�C��@�=q�������g
=C��                                    Bx���  �          @�\@�33��Q����B�\C�'�@�33��Q�Ǯ�EG�C�,�                                    Bx��b  �          @��@�����ff��{�k�
C�"�@�����ff����n�\C�(�                                    Bx��%  �          @�\@�Q���{�����j�\C���@�Q�����\)�m�C��
                                    Bx��3�  
�          @陚@�����=q�
=q��  C�)@����������33C�#�                                    Bx��BT  �          @�G�@��R����33���C���@��R��\)�z���G�C��=                                    Bx��P�  �          @�\@�G���
=�8Q�����C�7
@�G����R�9����ffC�AH                                    Bx��_�  
�          @�@���z��5���C���@����
�7
=���HC��f                                    Bx��nF  
�          @�@�����
=�5��  C���@������R�7
=��G�C��                                    Bx��|�  
�          @��
@�33��\)�\)���C�s3@�33��
=�������C�|)                                    Bx����  
Z          @��
@����z῞�R�
=C�
@����(���G��C�)                                    Bx���8  �          @��@�Q���
=��ff�"�HC�S3@�Q����R�����%C�XR                                    Bx����  �          @�(�@�Q���  ��H��  C���@�Q�����(���\)C�                                    Bx����  T          @��
@�  ��=q�:=q��{C��@�  �����;����C�ٚ                                    Bx���*  
�          @��H@�{����C33��ffC��)@�{��\)�Dz���C��                                    Bx����  
Z          @�\@��R��G��P����G�C���@��R�����Q��ԸRC��q                                    Bx���v  T          @�=q@��\��
=�HQ����HC�%@��\���R�I����=qC�1�                                    Bx���  
Z          @�(�@��H���\�-p����HC��@��H����.�R��Q�C��                                    Bx�� �  T          @��@�������5���RC�^�@��������7
=��{C�h�                                    Bx��h  
�          @�{@�p����\�(�����HC��3@�p���=q�*=q��(�C��)                                    Bx��  �          @�
=@�����(��,����(�C�w
@������
�.{��p�C��H                                    Bx��,�  
�          @�@��R�����*�H��p�C�j=@��R��Q��,(�����C�t{                                    Bx��;Z  
�          @�  @�������$z�����C��f@����z��%��  C��\                                    Bx��J   "          @�@��
���
�33���C�
@��
����z���ffC��                                    Bx��X�  "          @�\)@����z������\)C��@����(��
=q���\C�"�                                    Bx��gL  
�          @�
=@�z���
=��p��v{C��@�z����R�   �xz�C���                                    Bx��u�  
�          @�R@�����\�˅�E�C���@����=q��{�G\)C��                                    Bx����  �          @�
=@�p���  �����eG�C�Ǯ@�p������\)�g�C��                                    Bx���>  T          @�\)@�p�����   �x��C�l�@�p���\)�G��{\)C�s3                                    Bx����  
�          @�  @�\)���׿�\)�g�C�xR@�\)��Q����i�C�~�                                    Bx����  T          @�
=@�(���=q��\)�g�C��@�(���녿���j=qC�
                                    Bx���0  
Z          @�
=@������׿�z��M�C�33@�����Q��
=�Pz�C�8R                                    Bx����  
�          @�
=@���������F�\C��@������\)�I�C���                                    Bx���|  
�          @�R@�Q���\)����^{C�B�@�Q���\)���`z�C�G�                                    Bx���"  "          @�p�@�����R���pQ�C�3@����ff��Q��r�RC�R                                    Bx����            @��@�(����ÿ���_�
C�e@�(����׿��b=qC�j=                                    Bx��n  
�          @�p�@�Q����Ϳ��f=qC�R@�Q���z��{�h��C�q                                    Bx��  �          @�p�@��������R�y�C�0�@����33�   �{
=C�7
                                    Bx��%�  	�          @�p�@������ÿ�\)�j=qC�c�@������׿���l(�C�h�                                    Bx��4`  "          @�z�@�����R��
��G�C�7
@����ff�z���=qC�<)                                    Bx��C  "          @��@��H���
=��z�C�` @��H���Q����C�e                                    Bx��Q�  T          @�@��������{�g\)C�{@��������\)�iG�C�R                                    Bx��`R  
�          @�@�����ff�G��|��C�3@�����{��\�~�RC�
                                    Bx��n�  
�          @�ff@�{���R�G��{�
C�&f@�{��ff��\�}C�+�                                    Bx��}�  
�          @�{@�  �����(���33C�]q@�  ��G��p���(�C�b�                                    Bx���D  �          @�ff@�����H�(����RC���@�����H�������C��
                                    Bx����  �          @�p�@������0  ��(�C�Q�@������0����
=C�W
                                    Bx����  
�          @�Q�@�����׿�p��v{C���@����Q���R�w�C���                                    Bx���6  �          @�G�@�  �����\(�����C��@�  �����^�R���
C��                                    Bx����  T          @�=q@��\���׿=p���z�C�Y�@��\���׿@  ��
=C�Z�                                    Bx��Ղ  
�          @�  @����������$��C���@�����33�����%�C��{                                    Bx���(  
�          @�\)@�p���z´p��7
=C�Y�@�p���z῾�R�8Q�C�Z�                                    Bx����  "          @�\)@��
��(��ٙ��Q�C�C�@��
��(����H�R�HC�Ff                                    Bx��t  "          @�R@�\)������Q��1�C���@�\)������Q��2�HC��                                    Bx��  
�          @�
=@�  ��  �����B=qC��q@�  �����=q�C
=C���                                    Bx���  �          @�R@������ÿ�G��\)C���@������ÿ�G��(�C��3                                    Bx��-f  T          @�ff@������R�����b�HC�� @�����ff�����c�C��                                    Bx��<  T          @�  @�����p��u���HC��3@�����p��u��(�C��{                                    Bx��J�  
Z          @�\)@�\)��
=��(��UG�C��@�\)��
=��p��UC��                                    Bx��YX  ~          @�\)@�33��G���(��tQ�C�w
@�33��G���(��t��C�xR                                    Bx��g�  
�          @�R@�(����H��
=�O�
C�k�@�(����H��
=�PQ�C�l�                                    Bx��v�  
(          @�\)@�����������B=qC��@�����������B�\C��                                    Bx���J  �          @�\)@��
��{�   �xQ�C�<)@��
��{�   �xz�C�=q                                    Bx����  
�          @�Q�@�����z���H�Q�C��f@�����z���H�R{C�Ǯ                                    Bx����  
�          @�Q�@\��
=�O\)��  C��)@\��
=�O\)��  C��)                                    Bx���<  �          @��@�����ý��ͿB�\C�s3@�����ý��ͿB�\C�s3                                    Bx����  *          @�@�Q����R�#�
��  C��{@�Q����R�#�
���RC��{                                    Bx��Έ  T          @�ff@�
=��ff=�\)?�C���@�
=��ff=�\)?
=qC���                                    Bx���.  �          @�R@�ff��G����l��C��
@�ff��G����k�C��
                                    Bx����  
�          @�{@�33��33��R����C���@�33��33��R��  C���                                    Bx���z  
�          @�p�@��R���;�z����C��f@��R���;�z��{C��f                                    Bx��	   
�          @�{@�(����þ�(��U�C�33@�(����þ�(��R�\C�1�                                    Bx���  �          @�ff@�  ��  ��\�z=qC��)@�  ��  �   �vffC���                                    Bx��&l  �          @�ff@�{��녾�33�,(�C���@�{��녾�{�(Q�C���                                    Bx��5  T          @�ff@�=q����p��7�C�B�@�=q����Q��333C�AH                                    Bx��C�  T          @�\)@�
=���������B�\C��@�
=�����Ǯ�>{C��                                    Bx��R^  
�          @�  @��������p��333C�Z�@��������Q��.{C�Z�                                    Bx��a  
�          @��@�����ÿ������C�ٚ@�����ÿ�Q��\)C��
                                    Bx��o�  T          @�p�@��R��z�fff��  C��@��R��z�c�
����C�{                                    Bx��~P  "          @���@�p�����#�
���RC���@�p�����!G���33C���                                    Bx����  
�          @�p�@�����H�L����p�C��@����33�G����C�R                                    Bx����  
�          @�p�@�����J=q���
C���@�����E����C���                                    Bx���B  
�          @�z�@��\���\�\)���HC�5�@��\���\�
=q��ffC�4{                                    Bx����  �          @��@�G�����   �x��C��)@�G�������p  C���                                    Bx��ǎ  
�          @�ff@�G���
=��(��UC��3@�G���
=����K�C���                                    Bx���4  �          @�{@�����{�Ǯ�AG�C�˅@�����{��p��7
=C�˅                                    Bx����  �          @�R@��
��p��.{����C���@��
��p��������C���                                    