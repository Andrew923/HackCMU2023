CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230717000000_e20230717235959_p20230718021641_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-07-18T02:16:41.583Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-07-17T00:00:00.000Z   time_coverage_end         2023-07-17T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�Y@  "          Aa�@��
�9G�@*�HA.=qC���@��
�8��@333A6�HC��{                                    Bx�g�  T          Aap�@��
�9@ffA��C���@��
�9G�@�RA"=qC���                                    Bx�v�  �          A`��@����7\)@�
A�C��@����6�H@(�A Q�C��                                    Bx��2  "          A_33@�(��7\)@�\A�RC��
@�(��6�H@�HA�
C�޸                                    Bx���  "          A^�R@����6=q?�z�@��
C�#�@����5�@�\A
=C�+�                                    Bx��~  
�          A^{@�Q��733@{A\)C��H@�Q��6�H@
=A��C���                                    Bx��$  �          A\��@�z��7�@z�A
{C�e@�z��7
=@p�A�C�l�                                    Bx���  �          A\��@�R�7\)?���@�{C��f@�R�7
=?��RA��C��                                    Bx��p  
�          A]�@���4z�?��@�(�C�9�@���4  @�\A  C�AH                                    Bx��  �          A\��@�ff�6ff?\(�@dz�C�@�ff�6=q?�G�@��RC��                                    Bx��  "          A\Q�@��H�3�?�Q�@���C�xR@��H�3\)?���@��C�~�                                    Bx��b  "          A\  @��5�?���@���C���@��5��?��@�ffC��3                                    Bx�	  
�          A\(�@����1?��
@�{C���@����1p�?�Q�AC��q                                    Bx��  "          A\z�@��\�3\)?�Q�@��C�y�@��\�2�H?�{@��C���                                    Bx�&T  
�          A\Q�@�\�6�\?���@�Q�C��=@�\�6=q?�ff@θRC�Ф                                    Bx�4�  �          A\z�Az��/\)?n{@w�C��Az��/33?��@��C��=                                    Bx�C�  T          A\  Aff�0��?Y��@c�
C�7
Aff�0Q�?��\@���C�:�                                    Bx�RF  T          AZ{@����:�R>8Q�?=p�C��f@����:�R>�Q�?�  C���                                    Bx�`�  �          AX��@��3�?�33@�ffC��H@��3
=?�=q@ָRC���                                    Bx�o�  �          AX(�@����-?�(�A�\C���@����-G�@	��A�\C��3                                    Bx�~8  
�          AX��@��H�/\)?��@��
C��=@��H�.�H?���@�z�C���                                    Bx���  �          AX��@�\)�1��?G�@R�\C�l�@�\)�1G�?u@�=qC�q�                                    Bx���  T          AX(�@�ff�0��?Tz�@aG�C�l�@�ff�0��?��\@�=qC�q�                                    Bx��*  "          AW�@���/�?�ff@��C���@���/\)?��R@���C��)                                    Bx���  T          AW�@�G��0��?�\)@��C�*=@�G��0(�?�@�  C�33                                    Bx��v  �          AV{@�{�/\)?��A�C��@�{�.�R@z�A�RC�!H                                    Bx��  
�          AU��@��H�,��?��A�C��f@��H�,z�@�A�C��3                                    Bx���  
�          AT��@�{�,z�?�ff@��C��H@�{�,  ?�  @�\)C��=                                    Bx��h  �          AUp�@�\)�,��?���@�ffC��\@�\)�,(�?˅@ڏ\C��R                                    Bx�  T          AU�@�p��-?fff@w�C��)@�p��-p�?�{@�Q�C���                                    Bx��  �          AU�@��
�.�\>��H@�C�xR@��
�.ff?5@B�\C�|)                                    Bx�Z  �          AT��@�
=�/�?�@�RC��@�
=�/�?=p�@J�HC�"�                                    Bx�.   �          ATz�@�{�0  >�  ?�=qC��@�{�/�
>��@�\C�
=                                    Bx�<�  
�          AS�@�\�0Q�>���?���C�Ф@�\�0(�?�@33C��3                                    Bx�KL  �          AQ�@��H�/33?��@'
=C�u�@��H�.�H?Q�@g�C�z�                                    Bx�Y�  �          AP(�@ۅ�/\)?��@���C��@ۅ�.�H?У�@�C��                                    Bx�h�  T          AP��@����+�
?��RA(�C��{@����+33@{Az�C��                                    Bx�w>  �          AQ�@ᙚ�(��@@��AUp�C�� @ᙚ�'�
@O\)Af{C���                                    Bx���  �          AQG�@�\�$��@w
=A�\)C�>�@�\�#�@��HA�C�\)                                    Bx���  �          AR�R@���)p�@I��A]C�f@���(Q�@X��An�RC�q                                    Bx��0  "          AR�\@�  �(  @O\)AdQ�C�P�@�  �&�R@^�RAup�C�h�                                    Bx���  �          AS33@�\)�+�@)��A9��C�  @�\)�*ff@9��AK33C�{                                    Bx��|  "          AT  @�\�,Q�@Q�A%p�C��@�\�+�@(Q�A733C�.                                    Bx��"  �          AS\)@���-G�@�A(�C��@���,Q�@(�A*ffC��                                    Bx���  T          AS\)@�33�*{?�AC��=@�33�)G�@�A�
C���                                    Bx��n  �          AS33@�  �+33?��A�C���@�  �*ff@	��A{C���                                    Bx��  
�          AS33@�z��,Q�?�A Q�C�=q@�z��+�@
=A33C�L�                                    Bx�	�  T          AS33@��H�-�?�@�(�C��@��H�,Q�@�AG�C�"�                                    Bx�`  
�          AS�@�33�-p�?���@�p�C��@�33�,��@ffA=qC�!H                                    Bx�'  �          AR�H@�=q�-�?�  @��
C��@�=q�,Q�@�AC�)                                    Bx�5�  "          AR{@�R�+\)?�=q@�=qC�p�@�R�*�R?�{@��C�|)                                    Bx�DR  "          AS
=@���,��?���@ə�C�8R@���,Q�?޸R@�=qC�Ff                                    Bx�R�  
�          AQG�@�z��+\)?��@�  C�L�@�z��*�R?���@�G�C�Y�                                    Bx�a�  "          AO�
@�\�'�?��
@�=qC���@�\�'33?���@�33C��)                                    Bx�pD  T          AO
=@����#�
�u��\)C�Ф@����#�
>aG�?}p�C���                                    Bx�~�  T          AN=qA33�
=�u���C��A33�
==#�
>L��C���                                    Bx���  �          AN=qA��� z�>\)?(�C�t{A��� Q�>�(�?��C�xR                                    Bx��6  �          AM�@�Q��#\)�W
=�uC���@�Q��#\)=��
>�p�C��q                                    Bx���  T          AM�@��R�!��>\)?�RC��@��R�!p�>�G�?���C�!H                                    Bx���  �          AMG�@�\)� �׼��
��\)C�:�@�\)� ��>�\)?��\C�<)                                    Bx��(  �          AM��@�z��"ff>B�\?Y��C��@�z��"=q?   @�RC��{                                    Bx���  �          AQ��A�\��
?�33@�G�C��A�\�33?��H@�z�C��                                    Bx��t  T          AR�\A�
�#�?s33@��C�q�A�
�"�H?��\@���C�~�                                    Bx��  �          AR{A(��#
=?�@�\C��A(��"�R?Y��@l(�C���                                    Bx��  �          AQA\)�#33?(�@+�C�k�A\)�"�H?p��@��C�s3                                    Bx�f  T          AQ��A��"�H?(�@)��C�xRA��"�\?p��@�33C��                                     Bx�   
�          AR{A  �"�H?Tz�@g
=C���A  �"ff?�z�@�=qC��                                    Bx�.�  �          AS
=A{�"ff?Y��@l��C��A{�!�?�Q�@�p�C�Ф                                    Bx�=X  �          AR�HAG��"�\?�G�@�z�C��\AG��!�?���@�(�C��q                                    Bx�K�  �          AS33A���#�
?5@Dz�C���A���#\)?��@��HC���                                    Bx�Z�  �          AR�RA33�!�?Tz�@g
=C�HA33� z�?�
=@�(�C�                                    Bx�iJ  �          AR�HA�� ��?Tz�@g
=C�
=A�� z�?�
=@�z�C�
                                    Bx�w�  �          AR{A�H� Q�?h��@~{C�
=A�H��?�G�@�Q�C�R                                    Bx���  �          AQp�A�R��?Y��@n�RC��A�R�
=?��H@���C�"�                                    Bx��<  �          AR{A33� Q�?L��@`  C��A33��
?�z�@��HC�q                                    Bx���  "          AQAQ��
=?@  @S33C�L�AQ��ff?�\)@�z�C�Y�                                    Bx���  �          AQA��33?u@�{C�9�A���\?�=q@��C�H�                                    Bx��.  �          AQA
{��?u@��RC��A
{�z�?�=q@��\C���                                    Bx���  
�          AP��A
{��\?�G�@�(�C��HA
{���?�\)A  C���                                    Bx��z  �          AQG�A�>�=q?�z�C�U�A�p�?!G�@1�C�\)                                    Bx��   �          AR{A\)��?&ff@5C���A\)���?��\@��RC��q                                    Bx���  T          ARffA��>���?�C��\A�\)?333@B�\C��
                                    Bx�
l  �          AR{Az��Q�>�(�?��C��HAz��  ?L��@_\)C��=                                    Bx�  �          AR=qA���  ?��@Q�C���A����?k�@�Q�C��H                                    Bx�'�  T          AR=qA��ff?��@���C��A����?�@�ffC�#�                                    Bx�6^  "          AR�\A��33?���@�z�C��=A��{?޸R@�\C��                                    Bx�E  �          AQ�A����?�(�@�C��A�����?�{A=qC�"�                                    Bx�S�  �          AP��A��?˅@߮C��HA����?�p�A�C���                                    Bx�bP  �          AP��A  ���?�=q@�{C�HA  ��
?�(�A
=C��                                    Bx�p�  �          AO�
A����R?���@ᙚC��A�����@   A��C���                                    Bx��  
�          AP  A{��?�p�@���C���A{���?��AG�C��R                                    Bx��B  �          AP  A���
?�A�C�3A��ff@\)A�RC�7
                                    Bx���  �          AO�A(��ff?��@�p�C�H�A(����@(�A�C�l�                                    Bx���  �          AN�RA��?�\)A�C�FfA��Q�@�A"ffC�k�                                    Bx��4  �          AN�HA��G�@G�A�C�W
A���@�A-�C��                                     Bx���  
�          AO33A�R��@�A33C��RA�R�Q�@%A8Q�C�%                                    Bx�׀  �          AO
=A{��?��
@�C���A{�{@p�Ap�C��                                    Bx��&  
�          AN�HA{�Q�?�ff@�(�C���A{�
=?�p�A��C��                                    Bx���  �          AN=qA{��R?�\@�z�C��A{��@��AG�C�1�                                    Bx�r  �          AN=qA��?У�@��C�G�A��Q�@�
A\)C�k�                                    Bx�  
�          AL��A\)�33?��A�C��
A\)���@�A'�C��                                     Bx� �  �          ALQ�A
=��R?�{A{C��
A
=��@�
A&ffC��                                     Bx�/d  �          ALz�A
=q�(�?�Q�@�33C��HA
=q��R@��A�\C���                                    Bx�>
  
�          AK�
@�(��z�@AG�A[�
C�˅@�(���@_\)A~�HC�
=                                    Bx�L�  T          AK�A ����R@7
=APQ�C�AHA ���(�@U�As33C�}q                                    Bx�[V  "          AK33Az��33@1G�AIC�HAz����@N�RAlQ�C�=q                                    Bx�i�  T          AK
=A
=�(�@0  AH��C�� A
=�@N{Ak�C��)                                    Bx�x�  
�          ALQ�Aff�  @'�A=p�C��Aff���@EA`z�C�Y�                                    Bx��H  �          AM�A
=��R@:�HAR�\C���A
=�  @Z=qAv�\C���                                    Bx���  
�          AO
=@�{�!G�?Ǯ@�ffC�q@�{��@A�C�@                                     Bx���  
�          AN�\A   �!��?E�@Z�HC�0�A   � ��?��@�=qC�E                                    Bx��:  "          AO33@�\)�"ff?�G�@�\)C�{@�\)�!G�?Ǯ@��C�,�                                    Bx���  �          AO�
@��#�
?fff@\)C��@��"�H?��H@�ffC��R                                    Bx�І  �          AP(�@�G��$��?��@�Q�C���@�G��#
=@p�A(�C���                                    Bx��,  "          AO33@���&{?�z�@�C��@���$Q�@�RA�RC�+�                                    Bx���  "          AM�@��%?�ff@�C��@��$  @Q�AQ�C��                                    Bx��x  T          AM�@��"ff?У�@���C��\@�� ��@��A��C���                                    Bx�  �          ALz�A
�\��H?�z�A	C��A
�\���@�A/�C��                                     Bx��  T          AL��Az����?�p�A{C��)Az���@\)A3�
C�0�                                    Bx�(j  
�          AL��A�
�{@z�Az�C��HA�
��
@%A:�HC�R                                    Bx�7  T          AK�
A(���
@ffA(�C�P�A(����@(��A?�C���                                    Bx�E�  
�          AJ�HA=q�z�@�A
=C�\A=q�=q@'�A?
=C�Ff                                    Bx�T\  �          AK�A��\)@ffA*{C�P�A����@8��AR=qC��\                                    Bx�c  �          AM��A	p��  @�A.ffC�s3A	p��G�@>�RAV�\C��{                                    Bx�q�  "          AMG�AQ���@�
A&{C�7
AQ���\@7�AO
=C�u�                                    Bx��N  
Z          ALz�A=q��@\)A!��C���A=q��@3�
AK\)C�%                                    Bx���  �          ALQ�A�\���@
=qA�
C��)A�\�33@.�RAEC�8R                                    Bx���  
�          AL  A
=���@�A�C��A
=�ff@0��AH  C�Z�                                    Bx��@  �          AK�
Aff���@\)A"=qC��Aff�ff@4z�AL��C�G�                                    Bx���  �          AK�A	��@�A%p�C�� A	��
=@7
=AO�C��H                                    Bx�Ɍ  
�          AK�A��(�@�
A��C��A��@(Q�A?
=C�E                                    Bx��2  �          AJ�\AQ���\?�
=A��C�u�AQ��Q�@!G�A7�
C��\                                    Bx���  �          AJ�HA33���?���AQ�C��{A33�ff@=qA/33C�,�                                    Bx��~  �          AI�A���?�z�@�\C�s3A��  @  A%G�C��f                                    Bx�$  "          AHz�A���R?�(�@�z�C�G�A����@A,Q�C�|)                                    Bx��  
�          AH  @�\)�\)?���@��HC�AH@�\)���@�AG�C�k�                                    Bx�!p  �          AI�@��
�Q�?Tz�@r�\C�k�@��
�
=?��R@�G�C���                                    Bx�0  �          AJ=qA
=���?��@�C�L�A
=�\)?��A{C�s3                                    Bx�>�  �          AJ{A�H���?s33@��\C�FfA�H��?�{@��C�g�                                    Bx�Mb  "          AJ{Az��  ?aG�@�Q�C���Az���R?��@߮C���                                    Bx�\  �          AI��AG���?��@�(�C�ФAG��  ?�Q�A��C��q                                    Bx�j�  �          AHz�A���  ?�=q@�\)C���A����@{A#\)C�.                                    Bx�yT  
�          AHQ�A�
���?�=q@�Q�C���A�
�ff@\)A$��C��q                                    Bx���  �          AH(�A���{?��H@��
C�g�A����@Q�A/\)C��                                     Bx���  
�          AHz�A
=���?�\AC���A
=��\@�A3
=C��f                                    Bx��F  �          AI��A����?�{A�
C���A��
=@!�A9��C��                                    Bx���  T          AI�AG��z�?���AffC��3AG��@(Q�A@(�C�5�                                    Bx�  �          AJ=qA�����@ffA��C�� A����@1�AK
=C�'�                                    Bx��8  
Z          AJ�RAp����@
=qA��C���Ap��@5AO33C�:�                                    Bx���  �          AJ�\Aff��H@�A)�C�:�Aff��@@��A\Q�C���                                    Bx��  "          AJ�\A����\@"�\A9�C�0�A���
=@N{Al  C���                                    Bx��*  
Z          AJ{A{���@,��AEC�c�A{��@XQ�Ax��C�                                    Bx��  �          AJ�HA=q��@-p�AEp�C�NA=q�=q@Y��Ax��C���                                    Bx�v  "          AK33AG��@>�RAY�C�5�AG��@j�HA�z�C���                                    Bx�)  �          AK33A�@8��AR�HC�FfA���@eA���C��                                    Bx�7�  �          AJ=qA�����@<��AW�
C�B�A�����@i��A�=qC���                                    Bx�Fh  
�          AH��A���@UAw33C��A���H@���A���C�c�                                    Bx�U  
(          AH��A��33@6ffAQ��C�w
A��
=@c33A�p�C��                                     Bx�c�  �          AH��A���G�@$z�A<��C�1�A�����@R�\As
=C���                                    Bx�rZ  T          AHQ�AQ��(�@-p�AG�C�G�AQ��(�@[�A}�C��                                    Bx��   
�          AG33A���
=@p�A6=qC�p�A���\)@K�Al��C�Ф                                    Bx���  
�          AHQ�A��p�@&ffA@  C���A��	p�@Tz�Au�C�J=                                    Bx��L  "          AHQ�A	p��  @!G�A9��C�:�A	p��Q�@N�RAo\)C��                                     Bx���  	�          AG�Ap��=q@&ffA@��C��
Ap��
ff@U�Ax  C��q                                    Bx���  T          AG�
A���@!G�A:{C�l�A���@P��Ar=qC�Ф                                    Bx��>  �          AG\)A����H@$z�A>�\C�o\A���
�H@Tz�Av�HC��
                                    Bx���  
Z          AF=qA ���{@E�Af�\C��A ���	G�@tz�A�  C���                                    Bx��  �          AE�@�{��
@j�HA��C�!H@�{�=q@���A���C���                                    Bx��0  �          AE@�����
@}p�A���C���@�����@�ffA��HC�u�                                    Bx��  T          AEp�@��R�Q�@}p�A��C���@��R�ff@��RA���C�H�                                    Bx�|  
�          AE�@��(�@�  A���C�� @��{@�  A��C�B�                                    Bx�""  �          AD��@����z�@���A��
C�S3@����{@���A�=qC��)                                    Bx�0�  �          AD��@���
�R@��HA��C���@���Q�@�33A�(�C�ff                                    Bx�?n  �          AE�@��
���@a�A�33C���@��
�
=@���A��C��H                                    Bx�N  T          AD��@��	G�@��A�ffC��@��ff@�=qA���C���                                    Bx�\�  T          AEp�@������@��\A�ffC�]q@����ff@��HA���C�\                                    Bx�k`  T          AE@���	��@�ffA��C�  @����H@�
=A�C��
                                    Bx�z  �          AF=qA{�{@��A���C�� A{��p�@��HA�=qC�T{                                    Bx���  "          AEG�A�����@�z�A��C��A���p�@�33A�ffC���                                    Bx��R  �          AG\)@�
=�Q�@��\A��HC��=@�
=�G�@�33A��C�L�                                    Bx���  �          AH(�A��Q�@�G�A�z�C���A��G�@�=qA�\)C�|)                                    Bx���  T          AHQ�A33�Q�@�(�A�p�C�` A33��G�@�z�A��
C�9�                                    Bx��D  
�          AHQ�A��@�=qA�\)C���A����
@�=qAӅC��
                                    Bx���  
�          AHz�AQ����@�z�A�\)C�%AQ���(�@��
A�33C�"�                                    Bx���  �          AI�Aff���@��A�G�C�p�Aff��\@�z�A���C�q�                                    Bx��6  T          AJ=qAff� ��@�p�A�\)C�)Aff����@�AծC�                                    Bx���  
�          AIp�Aff�\)@�\)A�ffC�c�Aff��@�Q�AٮC�W
                                    Bx��  
�          AHQ�@��
�p�@�A�\)C�R@��
���@�{A�33C�&f                                    Bx�(  T          AHz�@�\)� ��@��A�z�C�aH@�\)��
=@�(�A�(�C�n                                    Bx�)�  �          AF=qA(��Q�@a�A���C��A(���@��
A��C��                                     Bx�8t  
Z          AC�A�\�
�R?��A�\C�3A�\��H@0  AP(�C�w
                                    Bx�G  T          ADQ�A	����@	��A!p�C��RA	���\)@?\)Aa�C�*=                                    Bx�U�  "          AF=qA=q�Q�?�(�@�=qC�u�A=q��@Q�A1�C��                                    Bx�df  �          AFff@����H?u@�{C�#�@���z�?�AffC�]q                                    Bx�s  �          AEp�A�\�33?�\)@�\)C��qA�\�z�@�\A(�C��H                                    Bx���  �          AEG�A��Q�?}p�@��C�U�A���?�A\)C��3                                    Bx��X  �          AEAff��H?�p�@�z�C���Aff��@��A3�C���                                    Bx���  �          AE�A	p��
=q@
=A�C�l�A	p���@@  A`��C��H                                    Bx���  �          AF�\AQ��
=q@   A:ffC�L�AQ��G�@Y��A~�HC��{                                    Bx��J  T          AEA�����@p�A8  C��fA����
@XQ�A~ffC�*=                                    Bx���  T          AD��A�
�  @Tz�Az�RC�\A�
���@��RA�(�C��H                                    Bx�ٖ  "          AD��@�\�
=@AG�Ac�C���@�\���@\)A��C�W
                                    Bx��<  "          AD��@��H�
=q@s�
A���C�q@��H��R@�\)A��C��                                    Bx���  
�          AD  A���@�z�A�(�C�5�A���\)@�Q�A�\)C�)                                    Bx��  
�          AD��AG��z�@�
=A�(�C�&fAG�����@�33AÙ�C�\                                    Bx�.  T          AD��A(��   @��\A�33C��{A(���\)@�{A�C���                                    Bx�"�  T          AD��A(���(�@���A���C�\A(���\@�33A�=qC��                                    Bx�1z  
�          AD��A����G�@�z�A�\)C�FfA����
=@�ffA�z�C�`                                     Bx�@   �          AD��AQ���33@��A�=qC�"�AQ���G�@�z�A��
C�7
                                    Bx�N�  
(          ADz�A
ff��(�@�(�A�
=C�ǮA
ff���@�A��C��                                    Bx�]l  T          AC�
A  ��  @�{A�(�C��A  �θR@�ffA�\)C�+�                                    Bx�l  
�          AC\)A z����@�=qA��
C��A z���p�@��RA�(�C���                                    Bx�z�  �          AB�\@�����@���A�=qC���@�������@�{A�=qC�|)                                    Bx��^  �          ABff@�����@�z�A�p�C���@������@��A�\)C��H                                    Bx��  
�          AB{@�p���@��
A���C�9�@�p���Q�@���A�Q�C�AH                                    Bx���  �          A@(�@�z���
=@�\)A�p�C�^�@�z���p�@�z�A��HC�c�                                    Bx��P  "          A@Q�A ����G�@��A�(�C��
A ����
=@��A���C�
=                                    Bx���  "          A@��A{����@�=qA��C�` A{���@�{A�Q�C��                                    Bx�Ҝ  �          AAp�A���=q@���A��C���A���{@�p�A�p�C��q                                    Bx��B  �          A@��Aff��\)@���A��RC��RAff�ۅ@�(�A�
=C��{                                    Bx���  "          A?�
A  ���@��A���C�fA  ���@�p�A��HC�9�                                    Bx���  �          A?\)A��陚@�\)A�C��HA��׮@��\A��C�                                    Bx�4  �          A?33A
=��Q�@��\A�  C���A
=��@�A͙�C�Ф                                    Bx��  �          A>ffAQ���ff@�(�A���C�~�AQ��ҏ\@�
=A���C�                                    Bx�*�  &          A=p�A ����z�@���A�z�C��
A ����{@�33A�{C�9�                                    Bx�9&  "          A>=qA33��@�z�A�  C��fA33��ff@�\)A�=qC��                                    Bx�G�  
�          A=G�A���
@�G�A��C�c�A���@�ffA��
C�z�                                    Bx�Vr            A=G�Aff��=q@�{A��\C��Aff�ָR@�=qA�{C�E                                    Bx�e  �          A=A=q��=q@��A�
=C��A=q��{@�ffAڸRC�H�                                    Bx�s�  T          A>ffA Q���{@���A�Q�C��A Q���\@��RAϮC�E                                    Bx��d  "          A>{@�z����\@��RA��
C���@�z���\)@�p�A�(�C��q                                    Bx��
  �          A>{@������@�G�A�p�C��)@����{@�  A�  C�Ǯ                                    Bx���  
�          A<��@��
��ff@qG�A�33C�Z�@��
����@�Q�A�ffC�b�                                    Bx��V  "          A<Q�@�z���  @�Q�A�\)C���@�z����@�
=A�(�C��H                                    Bx���  
(          A<��@�G�� Q�@n{A�G�C��@�G���
=@�\)A�\)C��                                    Bx�ˢ  
�          A<��@�����{@x��A�  C�AH@�����@�z�A�  C�U�                                    Bx��H  
�          A<��@�=q����@~�RA��C�XR@�=q���@�\)AǙ�C�t{                                    Bx���  
�          A<Q�@��
��
=@��HA���C��)@��
��@��A��C�                                      Bx���  
�          A<��@�Q����\@��A�G�C�^�@�Q���R@��AϮC���                                    Bx� :  
Z          A;�@��H����@�=qA�  C��3@��H��=q@�  A�z�C�f                                    Bx� �  �          A;�@��\��\)@���AŅC�� @��\��Q�@�=qA�(�C�q                                    Bx� #�  �          A;�@������@��
A��C��=@���߮@��AٮC��3                                    Bx� 2,  "          A<Q�@����{@���A�p�C��=@����(�@�z�B(�C��
                                    Bx� @�  T          A;�
@�����
=@�(�A�G�C��H@�����z�@ϮBQ�C�xR                                    Bx� Ox  �          A<(�@�z���=q@�A�(�C��{@�z���Q�@��B�C�XR                                    Bx� ^  
�          A;�A ����=q@�\)A��\C�` A ����33@��A���C��H                                    Bx� l�  �          A;33Aff��{@�{A�G�C��=Aff��
=@�33A�33C�O\                                    Bx� {j  �          A;�@�Q���
=@�  B �HC��@�Q���G�@�\B��C��\                                    Bx� �  	�          A=��@�����=q@�  A�Q�C�j=@�����ff@���Bz�C�T{                                    Bx� ��  
�          A=��@�ff��33@��A߮C��=@�ff��  @�  B��C�xR                                    Bx� �\  �          A=�@�z����@�Q�A�ffC���@�z���z�@�{B	\)C��)                                    Bx� �  T          A?
=@����\)@�33A�z�C���@���\@���B
=qC��                                    Bx� Ĩ  T          A@  @�33��  @�33A�=qC���@�33���@���Bp�C���                                    Bx� �N  �          AAp�@�����\)@У�B��C�Y�@�����
=@�ffBQ�C���                                    Bx� ��  
�          A@z�@�  �߮@ҏ\B��C��@�  ��
=@�Q�B��C�Ff                                    Bx� �  T          A@Q�@�  �߮@��BffC�f@�  ��
=@�  B�\C�E                                    Bx� �@  "          A@��@������@ٙ�B	�C��)@������H@���B ��C�H�                                    Bx�!�  
(          AA��@ۅ��@�z�B
=C���@ۅ��33@��
B"\)C�/\                                    Bx�!�  
�          AA@ٙ����@��B{C���@ٙ���@���B"��C�޸                                    Bx�!+2  T          AB=q@߮��{@��HB	=qC�)@߮��33@��\B �\C�s3                                    Bx�!9�  "          ABff@�=q��\@�(�B
�C�w
@�=q���@��B!=qC��q                                    Bx�!H~  
�          AAG�@�  ��ff@�ffB�HC��@�  ��(�@��B�C�z�                                    Bx�!W$  T          A9�@޸R��=q@�B�C�S3@޸R��  @��HB"(�C��R                                    Bx�!e�  
Z          A8(�A���\)@{AF�HC��A����@dz�A��RC���                                    Bx�!tp  T          A5G�A=q��ff@"�\AO�C�ǮA=q��\)@h��A��
C��{                                    Bx�!�  	�          A/�@�\)��\)?.{@c33C�l�@�\)�陚?�ffA�C��=                                    Bx�!��  
�          A/�
@�Q����þ\)�8Q�C�xR@�Q����R?�ff@�
=C��R                                    Bx�!�b  "          A.ff@����녿��333C��@������?+�@`��C��                                    Bx�!�  T          A-G�@�Q���G�>\)?=p�C��=@�Q���?�ff@��
C�!H                                    Bx�!��  �          A,z�@�ff��G��u���\C�˅@�ff��\)?n{@�{C��                                    Bx�!�T  �          A-p�A��\?!G�@Tz�C�q�A����?��HAffC�Ф                                    Bx�!��  T          A,��A�H�߮?�\@.�RC��qA�H�ڏ\?���AQ�C�3                                    Bx�!�  
�          A,  AG����þ���O\)C�|)AG���
=?s33@��HC��)                                    Bx�!�F  
�          A,  A z����
���\)C�7
A z���G�?��@�33C�b�                                    Bx�"�  T          A-��@�ff��(����
����C���@�ff�陚?��@���C���                                    Bx�"�  
�          A-G�Ap���{��=q����C�.Ap���z�?^�R@�33C�E                                    Bx�"$8  "          A.{A����H�
=�FffC���A����H?��@8Q�C��                                     Bx�"2�  "          A0z�A����
=�Dz�C���A��p�?\)@;�C��{                                    Bx�"A�  
V          A0��A=q���H�����C��A=q��<#�
=�\)C���                                    Bx�"P*  �          A0(�AG����
�}p����C��)AG���{>.{?fffC���                                    Bx�"^�  "          A.ffA�����Ϳ�z���{C�'�A����G���=q����C��                                     Bx�"mv  �          A.�\A�H���H������RC�w
A�H��=#�
>aG�C�J=                                    Bx�"|  
�          A.{A�R���
����{C�eA�R��33?(��@_\)C�n                                    Bx�"��  
Z          A-��A��أ׾��
�ٙ�C��\A���\)?J=q@�{C���                                    Bx�"�h  T          A.�\A  ��=q��(��G�C�� A  �ٙ�?333@k�C���                                    Bx�"�  
Z          A.�RA���θR���333C��
A���θR?
=q@7
=C��
                                    Bx�"��  �          A-��A\)�θR���(�C���A\)��{?�R@QG�C��H                                    Bx�"�Z  T          A,��A
{��\)�B�\��G�C��A
{��p�?fff@�Q�C��                                    Bx�"�   T          A*�\A	G���녾���Tz�C��
A	G��Ǯ?h��@�(�C���                                    Bx�"�  �          A*ffA�H�θR����>{C�@ A�H�θR?��@<(�C�@                                     Bx�"�L  "          A)��A�R���H��
=����C��A�R�׮������{C�>�                                    Bx�"��  
�          A)G�A�
���R��\)��Q�C�ٚA�
��z�?s33@�z�C�                                    Bx�#�  X          A*=qAz���33���
�ٙ�C���Az����?E�@��C���                                    Bx�#>  �          A)��A
=��z�>��?��C�q�A
=��  ?���@��C���                                    Bx�#+�  
�          A�R@�Q����\?���A\)C�&f@�Q���Q�@z�A^=qC���                                    Bx�#:�  "          A�H@�=q����?�{A!C��@�=q��p�@   A|��C�
=                                    Bx�#I0  �          A�@������?��A{C�@�����R@��Aq�C���                                    Bx�#W�  T          A(�@������H���Y��C���@�����G�?B�\@�=qC��                                     Bx�#f|  
�          A��@陚��  ��
=�,��C���@陚���>�@@  C��\                                    Bx�#u"  �          A
�R@�����?(��@���C�K�@������?��
A#
=C��{                                    Bx�#��  �          Az�@����G�?˅A+�
C��=@����{@
=A��HC���                                    Bx�#�n  T          A
�R@陚��?�33AG�C�b�@陚���@
=qAg�C�P�                                    Bx�#�  
�          A(�@����R?�G�A(�C��H@���p�@�AV�\C�Z�                                    Bx�#��  "          A�H@�����?�
=@�=qC��@����?��HAK33C�q�                                    Bx�#�`  "          Az�@�z���
=?s33@�Q�C�!H@�z���\)?��
A5��C��\                                    Bx�#�  T          Aff@�������?@  @�ffC�9�@������?���A ��C���                                    Bx�#۬  �          Aff@��H��  ?��@љ�C�N@��H��\)?�
=AB�HC��                                    Bx�#�R  
�          A�\@�p���>.{?��C���@�p���=q?��@���C��\                                    Bx�#��  
�          AQ�@�\)��z�@  ���
C���@�\)��ff=�G�?0��C��)                                    Bx�$�  �          A�@���������޸RC��@����
�u��  C�8R                                    Bx�$D  
�          A��@�33��Q쿴z��\)C�c�@�33����\�I��C��                                    Bx�$$�  
�          Ap�@�\)��������أ�C��{@�\)��\)�L�Ϳ��\C�B�                                    Bx�$3�  �          A  @�����
�(��y��C���@������>���?�z�C�t{                                    Bx�$B6  �          Ap�@�z����ÿ����C���@�z����;u��  C��f                                    Bx�$P�  
�          A
=@����G������D��C��3@����=q���
��{C��                                    Bx�$_�  "          A33@�����p��Y����=qC�Y�@�����\)�!G��~�RC���                                    Bx�$n(  
Z          A  @�Q���33�1G���=qC��@�Q���G������D(�C��3                                    Bx�$|�  �          A�H@����qG�������=qC�B�@����{�����C�
C�˅                                    Bx�$�t  �          A  @�p��~�R���H��C�)@�p���p������(�C�G�                                    Bx�$�  "          A�H@��|(���ff��C��q@�������33��{C��=                                    Bx�$��  T          A��@�p��5��G��D
=C���@�p����H���
�+��C���                                    Bx�$�f  r          A�@����(������_p�C�%@����r�\��p��E��C�@                                     Bx�$�  T          Ap�@.{���\���#�C�u�@.{�5��z��}��C���                                    Bx�$Բ  
�          A�@;���  � ����C�o\@;��AG����
�t\)C��                                    Bx�$�X  
�          A
ff@e��0������i�C�1�@e����
��z��JQ�C�~�                                    Bx�$��  T          A
=q@����N�R��p��U(�C��R@�����Q�����6(�C��R                                    Bx�% �  "          A	��@Y���������t33C�s3@Y���s�
����V{C��                                    Bx�%J  T          A
�H@Y����{�����\C�U�@Y���Fff���H�g�C���                                    Bx�%�  T          A
=@i���@������cC�H�@i�����
��G��C�C���                                    Bx�%,�  "          A	�@~�R�}p������E\)C��
@~�R��p���(��#(�C��\                                    Bx�%;<  T          A	�?��  ��
=W
C�9�?��q����
�pp�C�s3                                    Bx�%I�  T          A
�R@g
=�\���ȣ��M��C�+�@g
=���
��
=�+�C�                                      Bx�%X�  �          A  @�����{������C�f@�������������\C�:�                                    Bx�%g.  
�          A{@|(��(���=q�Rz�C�&f@|(��fff���R�6�\C�Ф                                    Bx�%u�  "          A�H@$z�>�G�����qA��@$z῎{��p��)C�G�                                    Bx�%�z  �          A��@333=��
��G�z�?�Q�@333���R����  C�                                    Bx�%�   
Z          AG�@Dz�Tz���p��C�u�@Dz��
=���H�x��C�>�                                    Bx�%��  �          @�\)@(��+����ǮC�P�@(�����
=G�C�!H                                    Bx�%�l  
�          @�?�녿!G���p��fC���?������(�W
C�t{                                    Bx�%�  "          @��R?˅����{8RC��)?˅�-p���G�\)C�33                                    Bx�%͸  T          A33?�ff��{���
��C��q?�ff�[���=q��\C��f                                    Bx�%�^  "          A
=@Q��<����=qC���@Q���{����X��C���                                    Bx�%�  
�          A�?�=q�Fff��33Q�C�=q?�=q��33�ٙ��WC��3                                    Bx�%��  "          A(�@z��(��   �3C�8R@z��e���p��s�C�f                                    Bx�&P  �          A{@{�@  � z�k�C��=@{�����ff��C�h�                                    Bx�&�  "          A�R@�G��.{��  �K�C�  @�G��z=q�����-=qC��{                                    Bx�&%�  "          A�@�=q��p��O\)��Q�C�~�@�=q��\)�	���k
=C��3                                    Bx�&4B  �          Az�@�=q��ff�XQ�����C�e@�=q�����G��v�HC��f                                    Bx�&B�  
^          A	@�33����K�����C���@�33���R���Y�C�p�                                    Bx�&Q�  
Z          A	�@����z��W���C��
@����\)�p��m�C�H�                                    Bx�&`4  �          A	p�@��������`  ��(�C��@�������
=�~ffC�ff                                    Bx�&n�  
(          A	��@ƸR�����k�����C���@ƸR��=q�'
=����C���                                    Bx�&}�  
�          A��@�33��Q������ffC�q�@�33��Q��<����p�C�Z�                                    Bx�&�&  �          A��@����Q��vff��Q�C�l�@����
=�,(�����C���                                    Bx�&��  
�          A��@�=q��������33C��=@�=q�����z=q��\)C�                                    Bx�&�r  
�          A��@�z���=q����
=C�  @�z���33��z���\)C�]q                                    Bx�&�  
�          A�@��\���
���H���C�U�@��\��p�������\C�Ff                                    Bx�&ƾ  
(          A  @�Q����H��Q��C�R@�Q���=q�y����C�Z�                                    Bx�&�d  
�          A33@�  ��\)��33�=qC�>�@�  ����i����=qC��=                                    Bx�&�
            A
�R@�������{�$ffC��@�����=q����C��H                                    Bx�&�  
�          A��@�����  �����-
=C�
@�����ff��33�C��\                                    Bx�'V  T          A@��\��p�����)�HC�5�@��\���
��=q��C���                                    Bx�'�  
�          A�@�G���Q������"
=C���@�G�������\��z�C��H                                    Bx�'�  
�          Ap�@�Q����\��(��&��C�\@�Q���Q���\)�Q�C���                                    Bx�'-H  "          A�\@��
�������&{C���@��
��  ��ff��ffC�˅                                    Bx�';�  T          AG�@��R������z����C��@��R������p����C�H                                    Bx�'J�  T          A��@�Q��������$�C�k�@�Q��������\���\C�c�                                    Bx�'Y:  "          A��@�  ��33�����$��C�l�@�  �����������
C�e                                    Bx�'g�  �          A�@�{��\)�����&z�C��@�{����������Q�C�"�                                    Bx�'v�  T          A��@�����  ����%=qC���@�����p��������C��                                    Bx�'�,  "          A=q@x�������z��.�HC���@x���˅���\���C���                                    Bx�'��  �          A�@k���Q���\)�L�\C�k�@k���
=���H�"��C�^�                                    Bx�'�x  "          A�@mp���ff��
=�(C��@mp�������H��(�C��=                                    Bx�'�  "          A�H@\(����
�����6�RC�Y�@\(��ָR��z��	�C��
                                    Bx�'��  
�          A\)@E���\��ff�MC�G�@E��33�����C��
                                    Bx�'�j  
�          A{@L(��a�� ���mC�
=@L(�������G��C{C���                                    Bx�'�  
�          Ap�@Vff�ٙ���
=qC���@Vff�fff���
�h  C�s3                                    Bx�'�  �          Aff@!녿޸R�	G���C���@!��j=q��ff�r��C�O\                                    Bx�'�\  �          A
=?��H?���{=qB�?��H�E��
=k�C�=q                                    Bx�(	  �          A�H?�?����{�B�R?��5�\).C�y�                                    Bx�(�  
�          A?�p�?h���	�p�A�  ?�p��������Q�C��H                                    Bx�(&N  �          A
=?��H?^�R�
�H8RA��H?��H���
=q\)C��                                    Bx�(4�  �          A{@33>\)�ff�@|(�@33�   ��Hu�C��                                    Bx�(C�  
�          A��@   �\)�p��C�+�@   �.�R�33�{C��                                    Bx�(R@  T          A
=?��R�k���\�qC���?��R�Fff��H#�C�Z�                                    Bx�(`�  
�          A��@zᾔz����W
C��@z��!���  C���                                    Bx�(o�  
�          A=q@Z=q�U���{�ep�C��)@Z=q�����ff�;\)C�/\                                    Bx�(~2  �          A�R@���������.{C�9�@�������{�\)C��R                                    Bx�(��  T          A�\@��
��{���
�;�\C���@��
����Q��(�C�S3                                    Bx�(�~  "          A�@G
=��{�����`�\C�H�@G
=��p��љ��1��C���                                    Bx�(�$  
�          A�@���>{�Q�\)C��H@����p���=q�X{C�>�                                    Bx�(��  
�          Ap�@U�(���z��{�C���@U��G�����R�HC�(�                                    Bx�(�p  T          A�\@p�׿��R���HC�,�@p���^�R����c��C���                                    Bx�(�  	�          A{@����
�H���R�ip�C��)@���������ff�I��C�Ǯ                                    Bx�(�  P          A\)@����8Q����W{C�q�@�����(���{�4�C��\                                    Bx�(�b  
�          A��@S33�O\)�ff�t��C���@S33�����HC�AH                                    Bx�)  �          A�@1G��w��=q�n�HC�Ф@1G���
=��
=�>p�C���                                    Bx�)�  
�          AG�@+��)���z�L�C���@+���z���(��[��C�                                    Bx�)T  �          Aff@Tz��N�R��H�q��C��)@Tz���(������E��C�p�                                    Bx�)-�  �          Az�@<(��H���
=�y=qC��3@<(���G����K�C�                                      Bx�)<�  �          A�@@���^{����pffC�u�@@����=q�����AC��q                                    Bx�)KF  
Z          A�@>{���R�����\  C��3@>{��ff��=q�*\)C��=                                    Bx�)Y�  T          A�H@/\)�E�  � C�Ǯ@/\)��z���
=�S��C�H                                    Bx�)h�  
�          A�@#�
��\)�33��C��@#�
��z��33�s
=C��R                                    Bx�)w8  
^          A"{@#33��z��ff\)C���@#33�p���  �|�HC�R                                    Bx�)��  
Z          Az�@G��#�
��B�C�
@G��:=q�z�Q�C��q                                    Bx�)��  
�          Aff@G�������3C�#�@G��{�z�=qC�G�                                    Bx�)�*  
�          A��?�p�?�����\B�?�p���{���Q�C�\)                                    Bx�)��  �          A�?\(�?�����B��?\(����ff¨�C�
                                    Bx�)�v  
�          A�R�z�@ ����H�B�W
�z����{«� CWT{                                    Bx�)�  �          A�Ϳ�Q�@S�
��
{B�\��Q�?�G�����=C                                      Bx�)��  "          Ap��.{@|���	���sp�B�
=�.{?�z����L�C�H                                    Bx�)�h  "          A��<��@h���	�u�HC�<��?���  ffCz�                                    Bx�)�  "          A���8Q�@Vff��
�|�
C�f�8Q�?��
�����C W
                                    Bx�*	�  
�          A\)�3�
@h������w{B�\)�3�
?���33��Cff                                    Bx�*Z            A�R�B�\@_\)�Q��x��C��B�\?�33�{�qCY�                                    Bx�*'   
�          AG��G
=@z����W
C^��G
=���
���C9�)                                    Bx�*5�  
�          A#33�e�@$z��\)�HCY��e�>�����C1�R                                    Bx�*DL  T          A%p��l(�@{���.C33�l(��#�
��R8RC4!H                                    Bx�*R�  T          A!��`��@\)��R�qC��`��=L���  p�C38R                                    Bx�*a�  "          A!��\(�@�\��
��CL��\(��#�
�Q�=qC6�R                                    Bx�*p>  �          A��X��@�p�G�Cc��X�ý�Q��{L�C5z�                                    Bx�*~�  
�          A!�U@��
aHC�R�U���z�\)C6:�                                    Bx�*��  
�          A"�R�j�H?�{���=qC��j�H�(��\){C=s3                                    Bx�*�0  �          A!���_\)?ٙ�����C���_\)�G��
=�\C@��                                    Bx�*��  �          A!�aG�?���{�Ck��aG�������RG�CEh�                                    Bx�*�|  T          A%p��g
=?�Q��G���CG��g
=��{���CE�                                    Bx�*�"  "          A%��Z�H?��
��W
C}q�Z�H�L���  ffCA.                                    Bx�*��  "          A'\)�y��?�{��aHC���y���s33��\�
CA�q                                    Bx�*�n  �          A'
=�r�\?��\�ffC!u��r�\����ff�)CG)                                    Bx�*�  "          A'
=�u�?�����C ��u���Q��=q�
CEW
                                    Bx�+�  
�          A&ff��  ?��
�(�G�C"=q��  ���\�(�Q�CE��                                    Bx�+`  �          A&�H���\?��\���u�C%�����\�������CH��                                    Bx�+   T          A'\)���R?��
��
B�C#����R���
��
B�CD�                                    Bx�+.�  "          A(�����?���ffL�C$p�������H�#�CG��                                    Bx�+=R  
�          A)G���(�?��\�
=�{C&���(��˅��CI�                                    Bx�+K�  �          A(Q���Q�?z�����RC,J=��Q�� ���ff�}�CM8R                                    Bx�+Z�  T          A$Q�����?(���\��C+ff������
=��(�CMz�                                    Bx�+iD  T          A#��P��?�
=�Q��
C��P�׿p���{�CD                                    Bx�+w�  T          A{��G�?�Q���H�\C#����G���G���RL�CEO\                                    Bx�+��  �          A$�����\?���p���C!� ���\��(��{CD��                                    Bx�+�6  
�          A%G����
?�����#�C%Q����
��z��  �~z�CE�)                                    Bx�+��  "          A%����ff?Ǯ�G��C�)��ff���\�ff��CA�                                    Bx�+��  �          A&ff��G�?����ffC!L���G������\z�CB�                                    Bx�+�(  
Z          A&�H���\?�\)����~�Cs3���\�z�H�
=��C@�q                                    Bx�+��  �          A&�R���\?�G�����}\)C޸���\�W
=�
=�fC?                                      Bx�+�t  T          A(Q��{�@Q��  ��CxR�{��
=q����C;�)                                    Bx�+�  T          A(����G�@"�\����w33C^���G������ff  C5W
                                    Bx�+��  "          A*=q��{@<(��33�o�RCxR��{>��R��\z�C0
=                                    Bx�,
f  T          A'�����@^�R�
=q�^=qC
=����?xQ��(��x��C(�=                                    Bx�,  �          A"=q��
=@Z=q����[�C���
=?�  ��\�vQ�C(�                                    Bx�,'�  �          A&ff��33@G
=�{�k��Cn��33?���=q�fC,�)                                    Bx�,6X  �          A"{�@��?�Q���HaHC+��@�׿=p����33CA�=                                    Bx�,D�  
�          A!p����?\��\L�C����ÿ��H�\)�CN�f                                    Bx�,S�  T          A!��z�?������ C���zῪ=q�p�W
CV�\                                    Bx�,bJ  �          A"=q��?����Q�Q�C���녿�
=�G�{CR)                                    Bx�,p�  
�          Aff�H��@   ���.C���H�ý��=q��C6@                                     Bx�,�  �          A����\)?�33����C
��\)��Q��G�=qCbp�                                    Bx�,�<  
�          A��>�������
¬W
C�:�>��,(�� ���=C��                                     Bx�,��  �          A�\@AG��
=�33�fC��@AG������R�S=qC��                                    Bx�,��  T          A�@H���,�����|�C��=@H��������(��K
=C�W
                                    Bx�,�.  "          A33@S�
������}��C��f@S�
�����ff�N=qC���                                    Bx�,��  
�          A�H@�G���ff�׮�7p�C��=@�G���p����R���C��                                    Bx�,�z  
�          A\)@�\)���R�����&p�C�n@�\)������Q����C�{                                    Bx�,�             A�@��
��z���
=�$p�C���@��
��\)��
=�C��)                                    Bx�,��  
�          A33@�Q���(������3p�C��@�Q����H���R�{C��)                                    Bx�-l  
�          A�@�  �h����  �Q=qC���@�  ���
�����"  C�XR                                    Bx�-  �          A�@��R�u��G��P{C��f@��R���\��(���\C���                                    Bx�- �  �          A
=@�{������G��JQ�C��)@�{��ff���H��
C�\)                                    Bx�-/^  �          A{@������
�У��1�C��@�����G�����G�C���                                    Bx�->  "          A\)@�ff���
��G����C���@�ff�׮�qG���(�C�z�                                    Bx�-L�  T          AQ�@����
=��ff���C�7
@���ڏ\�7
=���C��3                                    Bx�-[P  T          A��@��R���R������C�33@��R����5���RC���                                    Bx�-i�  T          @�\)@N�R�>{�{��'Q�C��
@N�R�~{�:=q��\C��\                                    Bx�-x�  �          @]p�@*�H��  ��z���\)C�J=@*�H���Ϳh���{�C���                                    Bx�-�B  �          @��R@�����?ǮA��RC��
@�����?�A�ffC��                                    Bx�-��  �          @���@�G�=u@�z�B=p�?L��@�G�?���@�{B2(�A�33                                    Bx�-��  �          @�(�@L��?L��@��Bj��Aap�@L��@@�ffBM=qB�                                    Bx�-�4  "          @�(�?h��?��@��B��)Bb��?h��@=p�@��RBg�B��                                    Bx�-��  T          @��?�
=��\@7
=B\�\C��H?�
=>�33@8��B_Q�A#�                                    Bx�-Ѐ  T          @�\)@o\)�!G�?�
=A��C��@o\)���R@
=A�p�C��R                                    Bx�-�&  �          @��@����>�ff@�p�C�xR@��
=q?���AF{C���                                    Bx�-��  �          @��\@����zᾞ�R�UC��@����z�>��R@Tz�C��                                    Bx�-�r  "          @��@��R�ff��
=���C���@��R�Q�>W
=@�RC�`                                     Bx�.  �          @�Q�@������!G����C���@����p�<��
>aG�C�(�                                    Bx�.�  "          @��@�33���B�\��  C�33@�33�  ���Ϳ}p�C���                                    Bx�.(d  P          @�(�@�  ����G���
C�  @�  ��
��
=�}p�C�(�                                    Bx�.7
  
�          @�{@����녿k����C�T{@����p���\)�&ffC��=                                    Bx�.E�  �          @��
@���\�xQ��ffC�"�@���R�����C�
C�E                                    Bx�.TV  
�          @�ff@�  ���H�����2�HC��
@�  ��R����  C�h�                                    Bx�.b�  �          @�\)@���녿���A�C�B�@�����&ff���RC��R                                    Bx�.q�  
�          @��H@��H�Q쿙���.�RC��)@��H��ÿ�\����C��                                     Bx�.�H  �          @�  @���
=q�Tz����HC���@���33������C�/\                                    Bx�.��  "          @��@��
�G���=q�!C�H@��
�\)��Q��S�
C�                                    Bx�.��  �          @��@�\)�z��p���
=C�1�@�\)�-p���p�����C�K�                                    Bx�.�:  T          @���@���� ����R��z�C���@����)����G����C�w
                                    Bx�.��  
�          @��@��H�33�%����HC�C�@��H�=p���\�r�\C���                                    Bx�.Ɇ  "          @�\)@������������C���@���{>��?�p�C�33                                    Bx�.�,  �          @��R@�p��2�\>\@���C���@�p��#33?���AL  C���                                    Bx�.��  
�          @��@�33�.�R?�@�C��
@�33�(�?�=qAn�RC�\)                                    Bx�.�x  T          @��H@��׿�������'C�Z�@�������w
=�p�C���                                    Bx�/  
�          @�{@�Q����=�G�?��C�&f@�Q���
?E�A�C��                                    Bx�/�  
�          @�\)@����!녿0����Q�C��@����'�=�G�?���C�<)                                    Bx�/!j  T          @��@Dz��G���
=��\)C��@Dz��G��������RC�L�                                    Bx�/0  P          @�\)@\(��
�H�n{�)�C���@\(��L���8������C��\                                    Bx�/>�  
�          @��R@����[�?��\A"�HC�~�@����<��@ ��A�
=C�}q                                    Bx�/M\  "          @ʏ\@�z��4z�@L(�A�G�C�\)@�z��(�@y��B��C�ٚ                                    Bx�/\  "          @ə�@x���\)@��RB)Q�C���@x�ÿ��@���BF�
C�,�                                    Bx�/j�  T          @�\)@	���u?^�RA�(�C��@	���&ff?�\)A�ffC���                                    Bx�/yN  T          @�R���?ٙ���ff�B�녿�Ϳz���(�§��Cb��                                    Bx�/��  �          @�(��L��@ �����\B�uÿL�;�����\¦��CJ)                                    Bx�/��  �          @�{��@\)��R��B������
���C7(�                                    Bx�/�@  "          @�
=�ff@R�\��
=�d  B�\�ff?�(���33C��                                    Bx�/��  
�          @�{��ff@-p���
=z�B�W
��ff>�Q���ff#�C(�                                    Bx�/  "          @�(���Q�?�=q��Q�C�=��Q쿣�
��\)z�CUc�                                    Bx�/�2  "          A�\�H��@>�R��p��k�\C��H��?�R���\C(�=                                    Bx�/��  
Z          A�
�i��@`���أ��T�
C{�i��?�  �����{z�C!�                                    Bx�/�~  �          A(��"�\@G������C
B��"�\��  �����C9�                                    Bx�/�$  �          A{�(��@
=��p�u�Cp��(�þ�p���{�HC<                                      Bx�0�  �          Az����@� Q�ǮC�׿����   �z��qCBJ=                                    Bx�0p  "          A��:�H?fff�p�£�RC8R�:�H��z���
=aHCv33                                    Bx�0)  �          Ap���G�?�  ����HC���G������\�C]{                                    Bx�07�  �          @�  ��?��
��  =qB��׽��#�
��ªG�C���                                    Bx�0Fb  �          @�
=>u@&ff���B�B��{>u>Ǯ��z�¬�Bk�                                    Bx�0U  �          @��
���?�����\)W
B�Q쿅���p��޸R¢CGh�                                    Bx�0c�  �          A �Ϳ:�H�B�\����¤�HCb#׿:�H�K���Q��{C��{                                    Bx�0rT  �          A(�>�ff�aG��
=§\)C���>�ff�hQ��\)�C���                                    Bx�0��  
�          Aff?}p���G��G�§� C�?}p��<���	G�(�C�@                                     Bx�0��  "          AG�>��þ������®B�C��f>����N�R��.C��=                                    Bx�0�F  �          AQ�>�33>���  ¯aHA�G�>�33�5�����C���                                    Bx�0��  T          A�<#�
?Q���HªB�
=<#�
���\)Q�C�
                                    Bx�0��  
�          A
=��?c�
��§z�B�����
=��R\)C��                                    Bx�0�8  "          A33���
?.{�ffª�HB�Ǯ���
�z��=qG�C�                                      Bx�0��  "          Ap���R�������¬L�C<�q��R�A����B�C�9�                                    Bx�0�  
�          A�R���H���R�=q¬�fCT)���H�P������C��3                                    Bx�0�*  T          A33�z�H��Q��=q§�C9J=�z�H�>�R�
=q�C{�q                                    Bx�1�  �          A(���\)�&ff��\¤B�CR33��\)�fff�\)33C|��                                    Bx�1v  T          A�H�@  �G��¦��Cb33�@  �mp��
{��C�L�                                    Bx�1"  
Z          A녿&ff�����\)\Cwk��&ff������v(�C��=                                    Bx�10�  T          Ap��(��
�H�(��C~B��(���33���
�eG�C�ff                                    Bx�1?h  
�          A\)��  �#�
���¥z�CT�ÿ�  �dz��	�u�C~c�                                    Bx�1N  
�          A=q��ff?B�\�
=C���ff�p��33� Cj��                                    Bx�1\�  	�          A��?333�G
=����C�P�?333��=q�ٙ��I�HC���                                    Bx�1kZ  �          A�?���R�\���R�C�{?������Ϯ�@��C���                                    Bx�1z   �          A
==u�z��
{�)C��3=u���R��{�a{C�T{                                    Bx�1��  
�          A�ÿ�G���\���(�Cs�)��G�����{�k  C�C�                                    Bx�1�L  �          A�:�H�Q��z��RC}\�:�H����Q��d�C�)                                    Bx�1��  �          Aff�   �7
=��\��C�  �   ������  �WG�C�s3                                    Bx�1��  "          A��>��\)����C��R>���Q���=q�_�C��f                                    Bx�1�>  T          A�R��\)��
=���R��C�AH��\)�����߮�l�C��                                    Bx�1��  T          @��Ϳ
=q�n{���¤Q�Co�)�
=q�X����33Q�C�q�                                    Bx�1��  "          @���?��\��(���(�z�C��H?��\��  ��33�]�HC��)                                    Bx�1�0  T          A	��?�Q��%�����C�xR?�Q���  �����Lz�C���                                    Bx�1��  "          A
=?���$z���=qC�H�?������{�]  C���                                    Bx�2|  �          A
=@,(���
=�	���oG�C�AH@,(��أ��أ��+��C��3                                    Bx�2"  T          A#\)@Y�������z��c��C�ٚ@Y����p������#  C�3                                    Bx�2)�  �          A#�
@��\�����=q�S
=C��f@��\������{�(�C��                                    Bx�28n  �          A$(�@�(����H����I��C�@�(����������
C��q                                    Bx�2G  "          A"�H@�
=��{���\�I33C�C�@�
=��R�����  C�+�                                    Bx�2U�  "          A�
@|��������G��M�C���@|����=q�����ffC���                                    Bx�2d`  �          A  @y����Q���ff�M�C�7
@y�����
��p���HC��
                                    Bx�2s  T          AG�@h���������MffC��@h����(��������C��                                     Bx�2��  T          A�@W
=����� z��c�\C��\@W
=��p������#�C��                                    Bx�2�R  T          Az�@�Q��������Y33C�Z�@�Q���p���p���
C���                                    Bx�2��  
�          A (�@��������p��H��C��)@����G����
��C���                                    Bx�2��  �          A!�@�{�`  �(��Z��C��H@�{��Q�����$��C���                                    Bx�2�D  
(          A$Q�@���R�\�  �`�\C�g�@����z��޸R�+33C�.                                    Bx�2��  T          A"ff@u�C�
���t{C��)@u�������
�:�RC��                                    Bx�2ِ  "          A\)@^�R�>�R�z��y�C���@^�R��{����>p�C���                                    Bx�2�6  �          A33@O\)��H���G�C�� @O\)���H����I33C�@                                     Bx�2��  �          AQ�@{��
�H��w
=C��=@{�������{�D�RC���                                    Bx�3�  �          A33@��R�0���33�h��C�"�@��R��(���\�5ffC�Ф                                    Bx�3(  �          Ap�@��H�G���p��_�HC�"�@��H�����ff�)��C�                                      Bx�3"�  
�          A@e����R�����T�C�33@e������33�=qC��R                                    Bx�31t  "          A   @B�\�ƸR��\)�1  C��@B�\�G������(�C�L�                                    Bx�3@  �          A"�R?�\�aG��  ǮC�XR?�\��G����H�A=qC�ٚ                                    Bx�3N�  
�          A
=�^�R@���
�u
=B�33�^�R?��H��
�B�u�                                    Bx�3]f  "          A��z���  �����i�Cq^��z����
�����#  Cz��                                    Bx�3l  "          A Q���p�@333�����@{C����p�>k���G��S
=C1�                                    Bx�3z�  �          A#
=��G�@aG���p��EC�{��G�?G��ff�a\)C+�q                                    Bx�3�X  "          A$  ����@�{��\)�<��C�\����?�p��
=�_�\C%
=                                    Bx�3��  �          A&�\���
@�����33�4  C����
@33�	���_Q�C�
                                    Bx�3��  T          A&{��
=@�p���  �8�HC�=��
=?�z��
{�`z�C                                     Bx�3�J  
�          A%G���z�@�p�����C�CJ=��z�?����Q��i�
C"�R                                    Bx�3��  "          Aff��G�@qG�� ���^=qC	��G�?^�R����C'�{                                    Bx�3Җ  
�          A   �\(�@�����C�\(��8Q��ǮC?�
                                    Bx�3�<  �          A#���
?�(��33�HCO\��
��{���CV�=                                    Bx�3��  �          A+��u�@\)��

=C��u������H.CD��                                    Bx�3��  T          A*{�tz�@ff��C}q�tz�}p��p�8RCB��                                    Bx�4.  T          A#��w
=@��\�  C� �w
=�Y����Rp�C@ff                                    Bx�4�  T          A%p��q�?�z���R�{C(��q녿����Q�G�CG�
                                    Bx�4*z  �          A"ff�s�
@ff�ff��RC^��s�
�Y���ff�C@�=                                    Bx�49   �          A��b�\?�\)�{�C^��b�\���\�
=�CGǮ                                    Bx�4G�  
�          A�
�b�\?�G������C���b�\����
=� CDk�                                    Bx�4Vl  �          A��S�
?�G��=q{C}q�S�
�����\�qCJ�q                                    Bx�4e  �          A
=�{�@-p��33�t��Cs3�{���p���C9T{                                    Bx�4s�  �          A��`��@{�z��}G�C�{�`�׿
=q��ffC<��                                    Bx�4�^  
�          A��N�R?޸R�z��)C���N�R��p���G�CL�)                                    Bx�4�  �          Ap��"�\?�����\33C���"�\�����=qCY#�                                    Bx�4��  �          AG��ff?s33��W
C�3�ff�p��(���C_L�                                    Bx�4�P  "          A�H��(�>�
=���£W
C!Ϳ�(��$z��
=�qCt�)                                    Bx�4��  �          A	�(�=����	p�«C*��(��333��#�C��
                                    Bx�4˜  �          Aff��=q�#�
���¢ǮC4���=q�>{�Q�Cu��                                    Bx�4�B  
�          A{���׾�  ���¥\C@\)�����L���33\Cz�)                                    Bx�4��  T          A{���;�{�(�¡��CA�3�����QG��=q��Cw��                                    Bx�4��  �          A�׿��������\ �=CE�Ό���\����
=Cw+�                                    Bx�54  
�          A33��ff�
=q���p�CG:��ff�dz��p���Cv��                                    Bx�5�  T          AQ����.{���B�C9�=����Tz���
L�Cq��                                    Bx�5#�  �          A��G���p��=q�C>B��G��g
=�33ffCp��                                    Bx�52&  T          A���녿(��G�  CD�R���tz�����ffCq�                                    Bx�5@�  
�          A{��(��
=��R�CD���(��tz��ff\)Cr��                                    Bx�5Or  �          A����H�333�Q�z�CJ5ÿ��H�s33��
�z�CuǮ                                    Bx�5^  �          A녿��H�E��(�¢�HCT^����H�|(��
�H.C|��                                    Bx�5l�  
�          AG��E��8Q��Q�§C_0��E��tz����{C�P�                                    Bx�5{d  "          A����R�W
=�(�W
CH���R�vff�
=�u�Co��                                    Bx�5�
  �          A33��ÿ=p���R8RCG{����s�
���y�RCp��                                    Bx�5��  �          A
=�Q�����R{CBh��Q��hQ��
=�~
=Co�H                                    Bx�5�V  T          A��p����H�{��CA����p��c�
��RQ�Cp�3                                    Bx�5��  �          A=q��
=�=p��ff�CH���
=�s33��{��Cs\                                    Bx�5Ģ  "          A�ÿǮ�Tz�����)COٚ�Ǯ�vff���|�Cw��                                    Bx�5�H  T          Aff��    �(�z�C4!H���?\)��
�Cp�)                                    Bx�5��  �          A\)��G�@#33�33�r{C�3��G������z�C:�                                     Bx�5�  T          A�\�~{@
�H�  �u
=C\)�~{�E���
{C?�                                    Bx�5�:  �          @ȣ��   ?�
=���HW
Cff�   �&ff��\)G�CE��                                    Bx�6�  �          @ָR?������
��33L�C��R?����\(���\)�T�\C�s3                                    Bx�6�  T          A�@љ���?�(�A(��C���@љ��j=q@9��A���C�e                                    Bx�6+,  "          A  @ə����
@&ffA�
=C��)@ə��]p�@���A�C��)                                    Bx�69�  �          @���@�(��o\)@��\A�
=C�0�@�(���@��
B!z�C��=                                    Bx�6Hx  �          @�@�=q�I��@�p�B��C��@�=q����@�\)B2��C�                                      Bx�6W  �          @�33@����L��@���BffC���@������@��
B0\)C��                                     Bx�6e�  �          @��H@����^�R@���B	�C���@��Ϳ�@�ffB.p�C�Q�                                    Bx�6tj  �          @�{@�=q�@��@���B0C�  @�=q�p��@�
=BN�HC�z�                                    Bx�6�  �          AG�@��
�U�@��HBp�C��@��
���@�p�B;�\C���                                    Bx�6��  �          A�
@���U�@�
=Bp�C���@����
=@��B3p�C��=                                    Bx�6�\  �          A�@�p��QG�@�Q�B
=C�s3@�p����\@��B=�HC�j=                                    Bx�6�  �          A Q�@����Dz�@���B
\)C��@������@���B&�
C��H                                    Bx�6��  �          A Q�@�G��H��@��HB�HC�8R@�G����@��
B2��C�l�                                    Bx�6�N  �          A (�@��H�J=q@�B�RC��@��H����@��B$(�C��                                    Bx�6��  �          AG�@��L��@�B(�C��\@����R@�Q�B"�RC��                                    Bx�6�  �          A ��@���_\)@��\B=qC�}q@�녿�ff@�Q�B#�\C�g�                                    Bx�6�@  �          @��H@��H�=p�@��\B�HC���@��H���@��\B!\)C��                                     Bx�7�  �          @��\@�p��Z�H@tz�A�C���@�p���33@�Q�B�
C��                                    Bx�7�  �          @��@�z��{�@l��A��C��@�z���@�=qB�HC��H                                    Bx�7$2  �          @��R@������@^�RA֏\C��=@����*�H@�BffC�q                                    Bx�72�  T          @��R@���\��@�(�A�{C�Ǯ@������@��B  C�b�                                    Bx�7A~  �          @�ff@���G�@��BG�C��q@����
=@�p�B"ffC�%                                    Bx�7P$  �          A��@�p��<(�@�{B	�RC�AH@�p���33@��B#
=C��
                                    Bx�7^�  �          A=q@��
�a�@n{A���C���@��
��\@�ffB�RC�p�                                    Bx�7mp  �          A�
@�p��l��@H��A���C�xR@�p����@�
=A�Q�C��                                    Bx�7|  �          A   @Ϯ�xQ�@&ffA��C���@Ϯ�0  @qG�A�\)C��f                                    Bx�7��  T          @�33@��
��=q@ffA��C�h�@��
�O\)@l(�A��\C��                                    Bx�7�b  �          @�(�@�{����@,(�A��C�y�@�{�7
=@z=qA��C��q                                    Bx�7�  �          @�
=@�����H@G�A��HC�l�@���a�@mp�A���C���                                    Bx�7��  �          A�H@�=q�>{@��Bz�C�~�@�=q��\)@��\B-{C��{                                    Bx�7�T  �          Aff@����
@�B�RC�@ @���\@��B/�RC�                                      Bx�7��  �          @��@�  �\)@�{B��C�]q@�  ����@��B,��C�                                      Bx�7�  �          @��H@���.�R@�B��C��@���s33@��HB+�RC�<)                                    Bx�7�F  T          @�G�@�z��0��@�33B
=C���@�z῀  @���B*�\C��3                                    Bx�7��  
�          @�z�@����@��HB��C��@��B�\@���B'(�C��                                    Bx�8�  �          @�33@���� ��@�33B�
C���@����p��@��RB��C�/\                                    Bx�88  "          @���@����\)@��
BC�b�@��ÿk�@�
=B%�C�3                                    Bx�8+�  
Z          @�(�@�녿�
=@��B%Q�C�3@�녾u@�ffB7  C��{                                    Bx�8:�  
�          @�=q@�=q�5@p��A�Q�C�xR@�=q��\)@�Q�B��C�3                                    Bx�8I*  �          @��H@����@��Bp�C��)@���E�@��B�HC�q                                    Bx�8W�  �          @�z�@����QG�@UAᙚC�#�@�����@�Q�Bp�C��                                    Bx�8fv  �          @�
=@�{��
=?��RAw�C�)@�{�~�R@e�A�C�@                                     Bx�8u  �          @��@�33��ff@{A�=qC�"�@�33�Tz�@vffA�p�C�{                                    Bx�8��  �          @���@��
���H@1G�A�=qC�"�@��
�G�@�33B
=C�t{                                    Bx�8�h  �          @���@�Q���=q@FffA�z�C��@�Q��0  @��B�C��R                                    Bx�8�  �          @��R@�G��r�\@X��A���C�ff@�G����@��B�C���                                    Bx�8��  "          @��H@�����@�{B�C��R@�녿J=q@�  B
=C�)                                    Bx�8�Z  �          @�ff@ƸR��ff@���B�C��q@ƸR<��
@���B��>8Q�                                    Bx�8�   �          @�@Å��
=@�(�B�\C��f@Å>���@���B�R@p��                                    Bx�8ۦ  "          @�=q@�ff��(�@��B�C�w
@�ff>W
=@��\B&(�@\)                                    Bx�8�L  T          @�G�@���ff@�G�B$ffC�q�@���@��B2ffC��\                                    Bx�8��  "          @ָR@����R@��B+p�C�/\@�>8Q�@���B6��@
�H                                    Bx�9�  T          @ٙ�@�
=��p�@��B-=qC�L�@�
=>aG�@�z�B7�
@*�H                                    Bx�9>  �          @�=q@��Ϳ���@�\)B8�HC�%@���?�R@��B<��A�                                    Bx�9$�  �          @ə�@�z��   @z=qB
=C�Ǯ@�z��@��B2G�C��
                                    Bx�93�  �          @˅@����ff@z�HB��C�t{@������@���B*�\C��{                                    Bx�9B0  �          @�33@�=q��R@h��B�HC���@�=q����@���B)�C�ff                                    Bx�9P�  �          @�Q�@4z�����?z�HA
=C��@4z����\@:=qA��C��                                    Bx�9_|  �          @���@.�R��(�?��
A��RC�H�@.�R�n�R@QG�B
=C��                                    Bx�9n"  �          @θR@a���(�@\)A���C��@a��s�
@q�BG�C�c�                                    Bx�9|�  �          @��H@aG���33?�ffA=��C�K�@aG����H@>�RA��C���                                    Bx�9�n  �          @�G�@x�����?   @��C�/\@x����{@
=qA��C��q                                    Bx�9�  �          @�{@H�����?��
A]C�+�@H����  @R�\A�\)C�o\                                    Bx�9��  �          @���@_\)��33?ǮA[33C��@_\)��\)@S�
A�C��                                    Bx�9�`  �          @�Q�@]p���\)?�ffA]��C���@]p����
@P  A�{C�0�                                    Bx�9�  �          @�z�@S33��ff?��HA��\C��q@S33����@S�
Bz�C��
                                    Bx�9Ԭ  �          @�33@S�
����xQ���
C�@ @S�
��33?��\A�HC�E                                    Bx�9�R  �          @�{@3�
��Q쿋����C��@3�
��  ?���A{C�q                                    Bx�9��  T          @��H@L(���33���
�H��C�Ф@L(���Q�?(��@�z�C��f                                    Bx�: �  �          @�z�@4z�����޸R��(�C��@4z���(�>W
=?���C�P�                                    Bx�:D  �          @�
=@}p��u@�
=B(  C��=@}p���33@��BW
=C�1�                                    Bx�:�  �          @��@�p��1G�@��\B�HC��q@�p����\@�  B)p�C��                                    Bx�:,�  �          @��@~{�Fff@���B;�
C�f@~{��ff@�(�B`�C��R                                    Bx�:;6  �          @���@h���;�@�{BS��C���@h�ÿ�R@�=qBvQ�C�(�                                    Bx�:I�  �          A33@�\)�?\)@��RB;�C���@�\)�B�\@��
BXz�C�p�                                    Bx�:X�  �          A�R@�  �
=q@ƸRB>�C��\@�  =�G�@�=qBMp�?�33                                    Bx�:g(  �          @�
=@p����\@��
Bd��C���@p��>�Q�@�p�Bu�\@���                                    Bx�:u�  �          @�?���@�G�B��qC��
?�@z�@�  B�BA\)                                    Bx�:�t  �          @�(�@  �Y��@�\)B��C��@  ?�Q�@�\B���B(�                                    Bx�:�  �          @�\@@  ��Q�@�=qB�Q�C�.@@  ?�p�@�=qB��A��                                    Bx�:��  �          @���@�ý�Q�@��B���C��H@��@=q@ۅB~�B5(�                                    Bx�:�f  �          @�\)@
=?h��@�G�B��3A���@
=@W
=@�G�BgG�B[�                                    Bx�:�  �          @�33?�?
=@�\B�=qA�33?�@J=q@�p�Bw\)Bj                                    Bx�:Ͳ  �          @��?��R?��\@��HB�=qB	�?��R@dz�@���Bq33B��3                                    Bx�:�X  �          @���?��H?��@�G�B�{B=q?��H@n�R@���BeG�B���                                    Bx�:��  �          @��?�\)>��@�G�B�W
Ab�H?�\)@1�@�
=Bv�HB`G�                                    Bx�:��  �          @��
@7�>�  @���B���@���@7�@#�
@���Bj{B'33                                    Bx�;J  T          @�@:�H?���@�
=B��A���@:�H@U@�BT�
BC{                                    Bx�;�  �          @陚@&ff?   @�p�B�ǮA-G�@&ff@0��@�33Bh��B:�
                                    Bx�;%�  �          @�33?c�
@��@�ffB��B��
?c�
@�G�@�Q�BQ�RB��q                                    Bx�;4<  T          @�{?z�H@(�@�z�B��{B�G�?z�H@��@��
BJ  B���                                    Bx�;B�  �          @�=q@
=<#�
@�(�B�L�>�  @
=@��@�{B|�B833                                    Bx�;Q�  �          @�=q@333��\)@�{B��C�
@333?���@���B���A��                                    Bx�;`.  T          @�z�@'��
=q@�z�B��
C�%@'�?�
=@���B�aHB�\                                    Bx�;n�  �          @�z�@(�?�
=@�ffB��A�R@(�@dz�@˅Bb�Bj�                                    Bx�;}z  �          @���?\?�ff@��HB�� B"\)?\@i��@ƸRBf=qB���                                    Bx�;�   T          @�G�@�\?h��@ҏ\B�A�@�\@G
=@�(�Bb�
BV�                                    Bx�;��  �          @�Q�?��@(�@�(�B�33BD��?��@��@��BA��B���                                    Bx�;�l  �          @�(�?��?���@�
=B��)BG(�?��@�Q�@�{BS��B��                                    Bx�;�  �          @�G�?��H@@ڏ\B�
=B\(�?��H@���@��RBM��B�p�                                    Bx�;Ƹ  �          @�\)?�Q�?��H@��
B��RB5(�?�Q�@n{@�ffB`��B��=                                    Bx�;�^  �          @�@  �Y��@�{B���C�� @  ?���@��B�.B��                                    Bx�;�  �          @�
=@)��>��@�ffB��R@U�@)��@�@�  Bp33B*                                      Bx�;�  �          @�z�@333>��R@ٙ�B��@�=q@333@'
=@���Bj�\B+��                                    Bx�<P  �          @�(�@9���n{@�B�G�C�&f@9��?���@˅B}z�AÙ�                                    Bx�<�  �          @�\)@AG���ff@�z�By�HC�h�@AG�?���@�(�By=qA�ff                                    Bx�<�  �          @�z�@O\)��@��
Bg33C�"�@O\)?�@���Bq=qA ��                                    Bx�<-B  �          @�(�@p�׿޸R@���BL�
C��\@p��=�@�B]ff?�G�                                    Bx�<;�  �          @���@n�R��\@�z�B:�HC�T{@n�R�.{@�\)BN�C���                                    Bx�<J�  �          @�p�@z�H�ff@���B/Q�C��H@z�H��@�\)BHp�C�+�                                    Bx�<Y4  �          @ƸR@\)��G�@�(�B4p�C�)@\)�.{@�
=BF�C�Ф                                    Bx�<g�  �          @�ff@k��!�@��RB-(�C��)@k��s33@�=qBN�C���                                    Bx�<v�  �          @�{@�p��,(�@&ffA���C��{@�p���33@W
=B=qC�4{                                    Bx�<�&  �          @���@r�\��@�A�33C��@r�\��  @=p�B\)C��f                                    Bx�<��  �          @��@J�H�}p�>aG�@   C�S3@J�H�i��?�ffA�Q�C�y�                                    Bx�<�r  T          @�Q�@O\)��p���(���G�C��@O\)���?��A7
=C�q�                                    Bx�<�  �          @��@[����׿L��� (�C���@[�����?G�@��C��)                                    Bx�<��  �          @��R@�  �{�=�Q�?fffC�s3@�  �j=q?�A[�C�n                                    Bx�<�d  �          @�=q@�{�Q�?p��A��C�� @�{�1G�?��RA���C��3                                    Bx�<�
  �          @��@����w�?��@��C�W
@����]p�?�=qA��\C��                                    Bx�<�  �          @��@��xQ콏\)�0��C�j=@��j�H?��RAC
=C�.                                    Bx�<�V  	1          @���@�33��Q�=p���33C���@�33����?(��@��
C��)                                    Bx�=�  �          @�{@�ff��33��33�EC��@�ff��ff?�p�A0��C�~�                                    Bx�=�  "          @Ӆ@��
��녿�ff�4��C���@��
��\)>��
@/\)C�*=                                    Bx�=&H  
�          @���@�����{����=qC���@�������?�\@�=qC�g�                                    Bx�=4�  T          @�@��
��  �޸R�qG�C��q@��
��=q�����
C��                                     Bx�=C�  T          @��@��������\)��Q�C�P�@������k����HC�7
                                    Bx�=R:  
�          @�\)@�����G��Q�����C�P�@�������?@  @���C�G�                                    Bx�=`�  "          @�ff@�����\)>�G�@vffC��q@����u?�=qA��C���                                    Bx�=o�  
�          @�@�\)�j=q@z�A��RC���@�\)�*�H@Z�HA�p�C��                                    Bx�=~,  "          @��
@�33��z�?�p�A@(�C���@�33�Y��@2�\A�Q�C�XR                                    Bx�=��  
�          @���@�Q����?���A	p�C��@�Q��w
=@#33A�G�C�
                                    Bx�=�x  
�          @��@�
=���\?��A ��C���@�
=�~{@ ��A��RC���                                    Bx�=�  �          @�\)@�{���>�@mp�C��=@�{���
@�
A�ffC�AH                                    Bx�=��  �          @�@�p���z���k�C���@�p�����?\A;�
C�P�                                    Bx�=�j  �          @���@�����\��p��{C�)@�����R>�@c33C��R                                    Bx�=�  
�          @��@��������z����C�޸@�����z�
=q��
=C��                                     Bx�=�  �          @��
@��R����
�H��p�C�j=@��R���\�?\)C�4{                                    Bx�=�\  �          @�G�@�����Ϳ�����\C�w
@����  ?�@���C�0�                                    Bx�>  �          @�33@�  ��G��Q���ffC�xR@�  ��G�?Q�@��C�w
                                    Bx�>�  T          @�  @�z����\@(�A��RC���@�z��Vff@`��A�C��                                    Bx�>N  T          @���@���33?�p�A<Q�C��@��u@;�A�  C�^�                                    Bx�>-�  
�          @�(�@�ff��z�>W
=?�C�o\@�ff����?޸RA[�C�y�                                    Bx�><�  
�          @�@�z�������33C�'�@�z���Q�?��A$Q�C��3                                    Bx�>K@  %          @陚@��\��(��
=q��
=C�+�@��\��G�?��
AffC�n                                    Bx�>Y�  �          @��@�������#�
����C�y�@����Q�?h��@�ffC���                                    Bx�>h�  "          @�  @��R��p���\����C��@��R���H?}p�@��\C�aH                                    Bx�>w2  �          @�@�p����
�
=q��33C�1�@�p�����?n{@�RC�g�                                    Bx�>��  "          @�=q@�������H���C��R@�����\>��
@#33C��H                                    Bx�>�~  
�          @�@�ff��G���\�^ffC���@�ff��zᾀ  ��Q�C�q�                                    Bx�>�$  
�          @�z�@�  ���R��
=�q�C��@�  �����(��U�C��f                                    Bx�>��  "          @�\)@��h���C�
�ȏ\C��@���ff��Q��X(�C�U�                                    Bx�>�p  
�          @�Q�@�33�n�R�ff��z�C���@�33��
=�@  ��{C�q                                    Bx�>�  "          @���@�\)�hQ��/\)��
=C�B�@�\)���\����4��C��)                                    Bx�>ݼ  �          @�z�@�z��^�R�i��� �RC��3@�z�����33���C��q                                    Bx�>�b  �          @�p�@�p����@�33B/��C��)@�p�?�@�{B3@�Q�                                    Bx�>�  �          @�z�@u��@�=qBWz�C�q@u?�  @�{BPffA�                                      Bx�?	�  	�          @�33@aG��Y��@�Q�Bb�C�5�@aG�?�G�@�\)Ba=qA\)                                    Bx�?T  
�          @���@j=q�aG�@�\)B]�C�>�@j=q?u@�
=B]{Ak�                                    Bx�?&�  T          @�Q�@����{@��B>�HC��{@��>�z�@��BH�@�Q�                                    Bx�?5�  �          @�p�@x�ÿ�{@���BGC�\)@x��>��
@��\BQ@�z�                                    Bx�?DF  �          @��
@�=q��p�@��B@�C��)@�=q>Ǯ@�(�BH=q@�ff                                    Bx�?R�  T          @�p�@���\)@���B:G�C�>�@�녿5@��BU�C��                                    Bx�?a�  "          @Ӆ@����=p�@��HB#ffC��@������@��BGG�C���                                    Bx�?p8  �          @�ff@����ff@�z�B-ffC��3@��׿J=q@�p�BH��C�j=                                    Bx�?~�            @�=q@��H�U@G�A�33C��\@��H��H@O\)A�33C���                                    Bx�?��  
�          @�@�ff���?�A#�C���@�ff�c�
@p�A�p�C�(�                                    Bx�?�*  
�          @�\)@��\��(�>�ff@uC��q@��\��  ?���A33C��f                                    Bx�?��  �          @��
@�����Q�>�\)@33C�h�@�����?�(�Ah��C�~�                                    Bx�?�v  "          @�z�@�33��(���33�:�HC�ff@�33��\)?�  A'
=C��R                                    Bx�?�  �          @�33@��R���?\AO
=C��@��R�Vff@0  A��C���                                    Bx�?��  T          @�Q�@�p��`��?ٙ�Ar�HC���@�p��1G�@/\)A�(�C��f                                    Bx�?�h  �          @θR@�G���(�?��HA,��C���@�G��aG�@{A��RC��                                     Bx�?�  
�          @Ӆ@�=q�.{?�p�AP(�C�s3@�=q�ff@�A�{C��                                    Bx�@�  T          @�z�@�G��@��?�ffA=qC�H�@�G�� ��?��HA�{C�K�                                    Bx�@Z  �          @�z�@�z�}p�@.�RA�{C��@�z�L��@:=qA��C���                                    Bx�@    "          @�\)@�=q���
@(�A��C���@�=q�B�\@(��A�G�C�                                    Bx�@.�  �          @���@��
���R?��A�\)C���@��
���
@�HA�C���                                    Bx�@=L  �          @ҏ\@�p��w�?�  A/
=C���@�p��QG�@�HA���C���                                    Bx�@K�  "          @Ϯ@���]p�?�(�A/\)C�K�@���8��@��A�p�C��{                                    Bx�@Z�  
�          @�33@�z��>{?\AV�\C�)@�z���@��A��C�Ǯ                                    Bx�@i>  �          @љ�@�z��~{?E�@�\)C�&f@�z��aG�?��RA�  C�˅                                    Bx�@w�  
�          @���@�
=����?L��@���C�33@�
=�c�
@�\A�
=C�޸                                    Bx�@��  �          @ҏ\@��R�~�R>�p�@Mp�C�K�@��R�j=q?�\)Ad  C�t{                                    Bx�@�0  	�          @�@�33���?B�\@�  C�%@�33�q�@�
A�{C���                                    Bx�@��  
�          @���@�(�����>Ǯ@R�\C��\@�(�����?�\Aq�C���                                    Bx�@�|  "          @ָR@��\��(�>�Q�@E�C�
@��\����?�Az�HC�8R                                    Bx�@�"  "          @׮@�Q���\)>��@`  C���@�Q����?��A�z�C���                                    Bx�@��  T          @�G�@��\��  >��@�C��q@��\��p�?�  Ao
=C��H                                    Bx�@�n  �          @׮@�G���
=�B�\��=qC��R@�G���G�?���A4Q�C�H�                                    Bx�@�  �          @أ�@�Q����׿�����C��q@�Q���{?��A�C�5�                                    Bx�@��  
�          @�Q�@��
��33�&ff��Q�C�L�@��
��G�?}p�A33C�t{                                    Bx�A
`  �          @��H@�\)�����G��l��C�y�@�\)����?���A"{C�Ф                                    Bx�A  "          @Ϯ@s�
��  ���R�U��C���@s�
��
=>#�
?���C�\                                    Bx�A'�  T          @�p�@��R�:=q>��R@J=qC�+�@��R�*=q?���AHQ�C�B�                                    Bx�A6R  �          @�=q@�G����?��@���C��
@�G��j=q?�A��C���                                    Bx�AD�  �          @��@�(��|��>�\)@(�C�33@�(��j�H?�  AU�C�9�                                    Bx�AS�  T          @�\)@�\)�n�R�.{�\C�AH@�\)�e?�  A�
C���                                    Bx�AbD  �          @�p�@��~�R�fff�C���@����>\@\(�C�AH                                    Bx�Ap�  �          @��
@�Q���  �����RC�c�@�Q��xQ�?�  A\)C��                                    Bx�A�  T          @��
@�  �~�R�z�����C�g�@�  �}p�?0��@�\)C�w
                                    Bx�A�6  �          @�Q�@�������Q��޸RC��R@������R?�\@��C��                                    Bx�A��  �          @׮@�G����;�Q��A�C���@�G�����?s33Ap�C�Ff                                    Bx�A��  �          @ָR@�p��|��>#�
?�=qC��@�p��mp�?�\)A<��C�˅                                    Bx�A�(  �          @�=q@�\)�_\)?aG�@��C�@�\)�C33?�A�(�C�y�                                    Bx�A��            @�(�@����p�@33A��HC��)@���p��@.{A�
=C���                                    Bx�A�t  �          @�\)@�=q��@r�\B
=C��@�=q?+�@p��B�@�                                    Bx�A�  �          @Ǯ@�
=�޸R@�A��C�33@�
=�fff@6ffA�G�C�P�                                    Bx�A��  �          @��H@�����
�����\C��{@���|��?�33A ��C�C�                                    Bx�Bf  �          @θR@�{�B�\?���AC33C�h�@�{��R@p�A��\C��H                                    Bx�B  �          @��@�
=�{?�ffA���C�h�@�
=�Ǯ@��A��C�aH                                    Bx�B �  �          @�
=@��H�	��?�G�A}G�C��q@��H��G�@A���C��)                                    Bx�B/X  �          @Ϯ@�p����?ǮA^=qC�{@�p��Ǯ@Q�A�C��                                     Bx�B=�  
�          @��@�G��G�?\A\Q�C�L�@�G��ٙ�@��A�ffC���                                    Bx�BL�  T          @ȣ�@�z���?���AB=qC��)@�z��{?�p�A�33C��                                    Bx�B[J  
�          @���@�z��{?�
=Atz�C�/\@�z��@ffA�
=C��
                                    Bx�Bi�  �          @ҏ\@���{@
=qA��C���@�녿��H@.�RA�{C��\                                    Bx�Bx�  �          @�  @�{�Q�@�\A�\)C��R@�{���@5�A���C�aH                                    Bx�B�<  �          @�
=@�=q��\@Q�A�=qC��R@�=q���H@=p�A�\)C���                                    Bx�B��  T          @�(�@���
=@�HA�=qC�5�@���\@AG�A��C�&f                                    Bx�B��  �          @�z�@�ff�
=@A�{C�G�@�ff��ff@<��AۮC�3                                    Bx�B�.  �          @��H@�33�p�@z�A��C���@�33��33@<��A�=qC�j=                                    Bx�B��  �          @��
@�
=�Q�@�RA�Q�C�7
@�
=����@6ffA�z�C��3                                    Bx�B�z  �          @�Q�@�z��ff@(�A�33C�<)@�z��=q@2�\A�33C�Ф                                    Bx�B�   T          @�  @�33��@ ��A��
C���@�33��p�@\)A�C��\                                    Bx�B��  �          @ƸR@�  �?�(�A�Q�C���@�  ��z�@ ��A���C���                                    Bx�B�l  �          @�z�@�{�?�Q�A��C��H@�{��z�@�RA�Q�C���                                    Bx�C  �          @���@�\)�33?�33A��
C�� @�\)���@�HA���C��H                                    Bx�C�  �          @�{@�G��   ?�\)A�z�C��@�G���{@Q�A��HC�                                      Bx�C(^  �          @���@�����G�?�p�A�ffC�8R@������@=qA��HC�q�                                    Bx�C7  �          @���@�����?�\)A�\)C�#�@���k�@\)A�C�<)                                    Bx�CE�  �          @��@��H��?��A'�C���@��H��Q�?�=qAup�C���                                    Bx�CTP  �          @��@�ff��(�?��HAbffC��@�ff���H?�p�A���C��f                                    Bx�Cb�  �          @�@�ff����?�Q�A`��C���@�ff����?�z�A��RC�1�                                    Bx�Cq�  �          @�p�@�G�����?��AP��C��@�G���{?޸RA��RC�Z�                                    Bx�C�B  �          @���@���У�?�{AT��C��q@����z�?�\A�Q�C�
=                                    Bx�C��  T          @�(�@�33����?�A_�C��@�33����?�
=A�33C�q�                                    Bx�C��  �          @�z�@��R��
=?���AXz�C�q�@��R���H?�A�33C��f                                    Bx�C�4  T          @�(�@����Ǯ?���A-��C�*=@�����?�G�An�RC�\                                    Bx�C��  �          @��
@�=q��(�?�\)A0(�C��q@�=q��=q?�  Al��C��                                    Bx�Cɀ  T          @�z�@�녿\?��HA=��C�\)@�녿���?���A|  C�e                                    Bx�C�&  �          @�33@�zῠ  ?Y��AC���@�z�s33?�A9��C�:�                                    Bx�C��  �          @��
@�����?Tz�A�HC�� @����G�?�
=A9�C���                                    Bx�C�r  �          @�(�@�33���
?n{A��C�U�@�33����?�=qAQC��3                                    Bx�D  �          @�ff@��R��  ?=p�@��
C��)@��R��p�?��A0��C���                                    Bx�D�  �          @�{@��R��ff>�@�p�C�k�@��R���?h��A��C�b�                                    Bx�D!d  �          @�z�@��
�Ǯ?Q�@�
=C�AH@��
��G�?�p�A@��C���                                    Bx�D0
  �          @�Q�@�  ��G�?O\)A ��C�P�@�  ��(�?���AA�C�                                    Bx�D>�  �          @�=q@�녿�ff?.{@�Q�C�=q@�녿��
?���A.{C��                                     Bx�DMV  �          @���@�G����H?L��@��HC��\@�G���?�
=A4��C�J=                                    Bx�D[�  �          @�G�@��\���?W
=@�ffC���@��\��ff?�
=A4��C��f                                    Bx�Dj�  �          @���@�������?^�RA��C�s3@�����ff?��HA:ffC��H                                    Bx�DyH  �          @��R@��ÿ�Q�?O\)@�G�C�+�@��ÿk�?�{A,Q�C��                                     Bx�D��  �          @�  @�33��{?J=q@��HC���@�33�W
=?���A$��C��                                    Bx�D��  �          @�ff@��׿���?L��@�{C�!H@��׿n{?���A*�HC�q�                                    Bx�D�:  �          @�p�@��ÿ��?0��@�C���@��ÿY��?xQ�A�\C��\                                    Bx�D��  �          @��@�Q�k�?��@�(�C�y�@�Q�5?Tz�A�HC�z�                                    Bx�D  �          @�=q@���Tz�>�@��C��q@���+�?333@��
C���                                    Bx�D�,  �          @�Q�@��W
=>��@��
C���@��0��?!G�@��C���                                    Bx�D��  �          @��@�p��O\)>���@S�
C��\@�p��0��?
=q@�ffC��                                    Bx�D�x  �          @�
=@��Ϳ@  >�p�@l(�C�5�@��Ϳ�R?\)@���C��                                     Bx�D�  �          @�=q@��ÿ�R>8Q�?�33C��\@��ÿ
=q>�33@fffC�33                                    Bx�E�  �          @���@��\�E�>�=q@2�\C�{@��\�(��>�@�33C���                                    Bx�Ej  �          @��
@��׿p��>�(�@��
C�*=@��׿J=q?+�@��
C��{                                    Bx�E)  �          @��@�\)�aG�>���@C33C�n@�\)�E�?�@�ffC�                                    Bx�E7�  �          @��\@���s33>�z�@@  C��@���W
=?
=q@��HC��=                                    Bx�EF\  �          @��H@��ÿTz�>B�\?�(�C��)@��ÿ=p�>��@�\)C�*=                                    Bx�EU  �          @��\@��ÿL��=��
?J=qC��H@��ÿ@  >�z�@AG�C�%                                    Bx�Ec�  �          @��H@��ÿO\)    ���
C���@��ÿG�>aG�@  C��q                                    Bx�ErN  �          @�33@�G��E��#�
��
=C��@�G��G�=L��?�C��R                                    Bx�E��  �          @�=q@�  �B�\���
�Tz�C��@�  �Q녽��Ϳ��C��                                    Bx�E��  �          @���@�
=�+���\����C��f@�
=�G����R�L(�C���                                    Bx�E�@  �          @��@��;�׿^�R��\C�}q@��Ϳ0�׿5��RC�W
                                    Bx�E��  �          @��R@��\�����\�,Q�C�*=@��\�J=q�W
=�G�C��\                                    Bx�E��  �          @��@�  �0�׿�ff�4  C�=q@�  �s33�Tz���C��q                                    Bx�E�2  �          @��\@�p��5��ff�5G�C��@�p��u�Q���C���                                    Bx�E��  �          @�z�@�
=�.{���G�
C�H�@�
=�xQ�p��� ��C��q                                    Bx�E�~  �          @��\@��;�33����hQ�C�\@��Ϳ0�׿�Q��O
=C�'�                                    Bx�E�$  �          @�Q�@��\���Ϳ��\�`(�C���@��\�:�H��\)�Dz�C��=                                    Bx�F�  �          @��\@���
=q�����q�C�  @���c�
��Q��N�RC�3                                    Bx�Fp  �          @�  @�=q�u�����^ffC��3@�=q�zῙ���K33C�޸                                    Bx�F"  �          @��@�����\����5p�C�k�@����G��@  �   C�!H                                    Bx�F0�  �          @�
=@�녿(������B�RC���@�녿aG��fff��RC�3                                    Bx�F?b  �          @�
=@�녿c�
�J=q�z�C��@�녿��ÿ���33C�
=                                    Bx�FN  �          @�\)@�Q쿱녾\)����C�E@�Q쿯\)>k�@ ��C�T{                                    Bx�F\�  �          @�@�(���>�33@w
=C��@�(����>��@�  C���                                    Bx�FkT  �          @���@�ff�(��
=q���C��H@�ff�=p���Q��|(�C���                                    Bx�Fy�  �          @��R@�ff�=p�����f�RC���@�ff��ff��ff�;�C�                                    Bx�F��  �          @���@��Ϳ�33����\)C�Z�@����33��  �b=qC�33                                    Bx�F�F  �          @�p�@�\)��G���z����
C���@�\)��33���
�dz�C�e                                    Bx�F��  �          @��@�G����R��Q���G�C�\)@�G��ٙ���ff��{C���                                    Bx�F��  �          @�{@���Q�����\)C��@��У׿\���\C�]q                                    Bx�F�8  T          @��@��׿(�����\)C�` @��׿��\���H��33C���                                    Bx�F��  �          @��H@�녿�ff��Q��Y�C�޸@�녿��ÿaG�� ��C�XR                                    Bx�F��  �          @�z�@��H�Y����Q�����C�@��H��Q쿕�T(�C��                                    Bx�F�*  �          @�G�@�ff��{��
�ڸRC��=@�ff�k�������C�#�                                    Bx�F��  �          @�
=@���+���Q����C��H@����=q��(���C�k�                                    Bx�Gv  �          @��\@�  ��녿���o�C�AH@�  ��
=�xQ��1��C��R                                    Bx�G  �          @���@�
=���
�:�H�ffC�
=@�
=��zᾣ�
�j�HC�L�                                    Bx�G)�  �          @���@��ÿ�(����H���C�w
@��ÿ�ff���Ϳ�
=C��                                    Bx�G8h  �          @�  @��H��G���z���C�K�@��H��=q����A�C��H                                    Bx�GG  �          @�
=@�G����Ϳ����`Q�C�H�@�G����ͿB�\���C��f                                    Bx�GU�  �          @���@��ÿ��R�����V{C���@��ÿ�p��8Q���
C��                                    Bx�GdZ  �          @�=q@���������R�o�
C��@�����H�W
=�!�C�^�                                    Bx�Gs   
�          @�p�@�  ��=q�(����C���@�  �������
�p��C���                                    Bx�G��  �          @�{@�  ��\)�������C���@�  ���H������C�8R                                    Bx�G�L  �          @�\)@�\)�
=�Ǯ��ffC�h�@�\)���>��?�p�C�AH                                    Bx�G��  �          @�Q�@�p���Q�����C�3@�p���\�\)��ffC��R                                    Bx�G��  �          @�ff@����p��z�H�.=qC��\@�����z���
=C���                                    Bx�G�>  �          @��@�����Q쿌���=p�C��@������5��(�C�޸                                    Bx�G��  �          @�=q@�  ������{�@  C��\@�  ���8Q���  C��                                     Bx�Gي  �          @�p�@��Ϳfff�����l��C��@��Ϳ��H��\)�>�HC�c�                                    Bx�G�0  �          @�@��R�fff��p��QC��@��R���xQ��%p�C���                                    Bx�G��  �          @��@�\)�0�׿�Q��J�\C�@ @�\)�u�z�H�'\)C���                                    Bx�H|  �          @�=q@��Ϳ0�׿�\)�B{C�/\@��Ϳp�׿k���HC��3                                    Bx�H"  T          @�33@�z��\�����o�C�0�@�z�Q녿�(��R�\C�q�                                    Bx�H"�  T          @�=q@�녾�G���{��z�C�� @�녿Q녿����}G�C�j=                                    Bx�H1n  �          @��
@�������\)��G�C��=@����5�������\C�*=                                    Bx�H@  �          @��@�z�(���\)�mC���@�z�k���Q��L��C��                                    Bx�HN�  �          @�z�@�
=��׿�  �U�C�o\@�
=�@  �����<  C��                                    Bx�H]`  �          @��
@�Q�\)�W
=���C���@�Q�=p��0����C��)                                    Bx�Hl  �          @�(�@�G��=p��0�����
C�@�G��aG����H����C�B�                                    Bx�Hz�  �          @�@����\�=p���33C�C�@���+�������HC�j=                                    Bx�H�R  �          @���@��H�Ǯ�0���陚C��f@��H�
=q���ÅC�R                                    Bx�H��  �          @�p�@�����
�E���\C�E@���   �+����
C�\)                                    Bx�H��  �          @�p�@�33��
=�333���
C���@�33�녿z��ÅC���                                    Bx�H�D  �          @��@�{�녾���33C��q@�{�+����
�S�
C��H                                    Bx�H��  �          @�
=@��;��0����C�p�@��Ϳ!G��\)��33C���                                    Bx�HҐ  �          @�ff@�(��
=q�&ff��33C�'�@�(��+���\���HC�p�                                    Bx�H�6  T          @�\)@�p����!G���33C�9�@�p��(�þ��H��z�C��=                                    Bx�H��  T          @���@��R�����\��=qC��q@��R�333��Q��n�RC�W
                                    Bx�H��  T          @�G�@���
=�����Q�C��3@���+���=q�0  C���                                    Bx�I(  �          @�G�@�  �\)�Ǯ����C��@�  �!G���  �&ffC��R                                    Bx�I�  �          @�G�@�Q��(���Q��n{C��H@�Q�   ��  �%C�c�                                    Bx�I*t  �          @���@�����\�~{C�9�@���(���  �'
=C��R                                    Bx�I9  �          @���@�\)���H�������
C�k�@�\)�녾�=q�5C�                                    Bx�IG�  �          @��@��R��ff�\�{�C���@��R������0��C�AH                                    Bx�IVf  �          @�  @�ff����(���  C�7
@�ff��R�����HQ�C�Ǯ                                    Bx�Ie  �          @�
=@���������{C�Ф@���8Q��
=���C�:�                                    Bx�Is�  �          @�\)@�p��
=q����ə�C�,�@�p��(�þ�����
C���                                    Bx�I�X  �          @��@����=q������C�P�@����zᾅ��0  C��q                                    Bx�I��  �          @�Q�@�=q��\)��G����HC�˅@�=q��
=�����C�xR                                    Bx�I��  �          @���@��
��33��G����C���@��
��(��.{���
C��R                                    Bx�I�J  �          @�  @���=�?���C�:�@���\?�@�p�C��{                                    Bx�I��  �          @�\)@����(�=#�
>�ffC�|)@����z�>�@�
=C���                                    Bx�I˖  �          @���@�녿�33=�Q�?�G�C���@�녿�=q?�\@��C��                                    Bx�I�<  �          @���@�����\    �#�
C�33@��Ϳ��R>�ff@��C�p�                                    Bx�I��  �          @��R@�ff�'
=?G�A�
C�u�@�ff�Q�?�=qAb�RC���                                    Bx�I��  �          @���@��H��þ����g
=C��@��H�
=q>\)?�G�C���                                    Bx�J.  �          @���@��׿��
���ƸRC�7
@��׿\�ff��{C��
                                    Bx�J�  �          @��H@�33�@  �*=q��p�C��q@�33��ff���θRC�y�                                    Bx�J#z  �          @���@��
�=p��\)��z�C���@��
��G��G��\C��                                    Bx�J2   �          @�Q�@�=q�ٙ������4Q�C��)@�=q��녿.{��(�C���                                    Bx�J@�  �          @�  @��
�n{�
=���C��3@��
���
=���RC���                                    Bx�JOl  T          @���@��������Ϳ��
C��q@���
=>��@��HC�&f                                    Bx�J^  �          @�Q�@��\�ff���Ϳ��
C�@��\�z�>�33@h��C���                                    Bx�Jl�  �          @���@��׿Ǯ����
=C��@��׿�\)�\)��Q�C�p�                                    Bx�J{^  �          @��R@�����  ���R�S�
C�^�@�������L�;��C�,�                                    Bx�J�  �          @�p�@�
=��33����.{C�|)@�
=��
==#�
>�ffC�]q                                    Bx�J��  �          @��R@����#�
?G�A�
C��f@����?�ffA\(�C���                                    Bx�J�P  �          @��\@���\)?�Q�AM�C���@����?�
=A��HC�1�                                    Bx�J��  �          @���@�  �$z�?��A=�C�'�@�  ��?���A�z�C���                                    Bx�JĜ  �          @�@���E�?���A<  C�8R@���2�\?�(�A�C���                                    Bx�J�B  �          @��@����N�R?���A3�
C��=@����<(�?��HA�\)C��\                                    Bx�J��  �          @�ff@����5?�=qA7\)C��
@����#�
?��A�(�C�G�                                    Bx�J��  T          @��
@��R�0  ?�AH��C�*=@��R�p�?ٙ�A�  C���                                    Bx�J�4  T          @��\@��2�\?�  A,��C��@��!�?��A���C�!H                                    Bx�K�  
�          @��@�G��+�?n{A z�C��R@�G��(�?��HA|z�C���                                    Bx�K�  �          @���@��   ?�p�A��C�E@��	��?��HA�G�C��                                    Bx�K+&  �          @�z�@���8Q�?��Aa��C�J=@���#33?�{A��C�Ф                                    Bx�K9�  �          @�(�@�=q�2�\?�Av�RC��
@�=q���?���A�\)C�=q                                    Bx�KHr  �          @�Q�@�33�;�?5@�p�C�@�33�.�R?��
AW\)C��=                                    Bx�KW  �          @�ff@����?\)?�\@�C�<)@����5�?���A:=qC���                                    Bx�Ke�  �          @�ff@�Q��B�\>��
@W
=C��)@�Q��:�H?h��A�RC���                                    Bx�Ktd  �          @�@����;�>��@0��C��H@����4z�?Tz�Ap�C��                                    Bx�K�
  �          @��@��H�*=q=��
?W
=C��3@��H�%?��@�
=C�C�                                    Bx�K��  �          @��H@�=q�C�
��33�s�
C�Y�@�=q�Dz�>�  @.{C�N                                    Bx�K�V  �          @��\@�(��%��(����
C�j=@�(��'�=��
?c�
C�AH                                    Bx�K��  �          @�G�@��333�.{����C���@��8Q������C�u�                                    Bx�K��  �          @�  @��(�ÿp���$��C��q@��1G������\)C�                                      Bx�K�H  
�          @���@�\)�'��k�� ��C��
@�\)�0  �������HC�=q                                    Bx�K��  �          @�\)@�\)�!G��k��!p�C�P�@�\)�)����
=���C���                                    Bx�K�  �          @�\)@���.�R�fff�\)C��
@���6ff��p�����C�e                                    Bx�K�:  �          @���@�(��1G��p���$��C��3@�(��9���������HC�:�                                    Bx�L�  �          @�Q�@�ff�)���c�
���C���@�ff�1G��\��(�C�\                                    Bx�L�  �          @���@�z��333�Tz��C��
@�z��:=q��z��J�HC�9�                                    Bx�L$,  �          @��@�33�E��ff���
C�Q�@�33�G�>�?�C�0�                                    Bx�L2�  �          @�z�@����L�Ϳ#�
�ۅC���@����QG�����33C�G�                                    Bx�LAx  �          @�{@�G��p�����33C��@�G��.{��
=�IG�C���                                    Bx�LP  �          @��R@�  �)����=q����C���@�  �9������3�C���                                    Bx�L^�  �          @�{@�\)�*=q�����C��@�\)�8�ÿ��\�-�C��{                                    Bx�Lmj  �          @��@�ff�<�Ϳ��
��
=C�s3@�ff�K��xQ��&{C�k�                                    Bx�L|  �          @��@qG��R�\����Q�C�o\@qG��e�����_
=C�5�                                    Bx�L��  �          @��@�  �2�\��33��(�C�\)@�  �C33��\)�AG�C�1�                                    Bx�L�\  �          @��@����.{�(���Q�C���@����Dz��z���p�C�O\                                    Bx�L�  �          @�G�@QG��>{�8�����C��H@QG��\(���
��p�C��                                    Bx�L��  �          @�(�@e��>{�,(����C�,�@e��Y��������C�:�                                    Bx�L�N  �          @�ff@Z�H�A��,(����C�<)@Z�H�]p��
=��33C�T{                                    Bx�L��  �          @�@B�\�a��1G����C�^�@B�\�~{�ff���C��q                                    Bx�L�  �          @�ff@H���Dz��H���G�C���@H���e��#33���C���                                    Bx�L�@  �          @�33@g
=�0���N�R��\C�P�@g
=�R�\�,(�����C��)                                    Bx�L��  �          @�@aG��8Q��Z=q�p�C�]q@aG��\(��6ff���HC��q                                    Bx�M�  �          @��@n�R���j=q�!�C�1�@n�R�.�R�O\)���C��=                                    Bx�M2  �          @��
@e����z=q�0C�b�@e��\)�a��  C���                                    Bx�M+�  �          @��H@n{�\�y���0z�C��@n{�
�H�dz��C��)                                    Bx�M:~  �          @��R@aG���
=�u�2(�C�:�@aG��z��_\)��\C�P�                                    Bx�MI$  �          @��@@  �/\)�^{�!�HC�Ǯ@@  �S33�<����HC�&f                                    Bx�MW�  �          @��@7��O\)�J=q�\)C���@7��o\)�#�
���C���                                    Bx�Mfp  �          @�z�@3�
�C33�Z=q���C�Z�@3�
�e��5����C��                                    Bx�Mu  �          @���@I���\)�Z�H�!z�C��)@I���A��<����
C��                                    Bx�M��  �          @���@>�R� ���e�*��C���@>�R�E��G��(�C�\                                    Bx�M�b  �          @��@C33��H�e��*�\C��f@C33�?\)�HQ��  C�˅                                    Bx�M�  �          @��@XQ��33�g��*(�C�\)@XQ��(Q��N�R�G�C�\                                    Bx�M��  �          @��\@E����  �B�C��R@E��
�j�H�.Q�C���                                    Bx�M�T  �          @���@>�R��Q��xQ��>  C�u�@>�R�#33�`���'�C���                                    Bx�M��  �          @��@E��Ǯ�|(��C(�C��\@E��(��hQ��/�C�N                                    Bx�M۠  T          @�(�@k���G��O\)�  C���@k����
�?\)���C�)                                    Bx�M�F  �          @�Q�@~{��
�(���Q�C�O\@~{��ÿ������C��H                                    Bx�M��  �          @�ff@g�������C�j=@g��.{���H���HC���                                    Bx�N�  �          @���@N{�z��C33�\)C���@N{�"�\�+���RC��q                                    Bx�N8  �          @�{?�ff��\)��Q�  C�H�?�ff�
=��
=�n�\C�"�                                    Bx�N$�  �          @���?��ÿ�p����\Q�C���?����\)�����p��C�AH                                    Bx�N3�  �          @�
=��=q�0�����8RCTaH��=q��(���=q�Ci�f                                    Bx�NB*  �          @�  �(��8Q�����s�CDW
�(���
=��z��e  CRT{                                    Bx�NP�  �          @�{�����!G����Q�CK��������33��  {C`
=                                    Bx�N_v  �          @�G��&ff�5����CcW
�&ff��G���\)u�Cv��                                    Bx�Nn  �          @�=q�!G���z���33aHCu�!G�����=q�C}�3                                    Bx�N|�  �          @�ff���
��ff��G��=Cf�����
�z�����sp�Cq#�                                    Bx�N�h  �          @��ÿ�녿�����Rp�CX�f��녿������Ch
                                    Bx�N�  �          @��þ��>aG����R©�)C@ ��녿z���{¤z�Ck�                                    Bx�N��  T          @����8Q�?�  ��(��B��H�8Q�>W
=��\)£��C$+�                                    Bx�N�Z  �          @��#�
�(����¦G�C�g��#�
�������R�
C��)                                    Bx�N�   �          @����+��(���p� .C^@ �+���������  Ct&f                                    Bx�NԦ  �          @�녿   ���R��(��qCx5ÿ   �   ��(�B�C�                                    Bx�N�L  �          @�G��+����H��33��Cq5ÿ+����H�����C{8R                                    Bx�N��  �          @��R�(�ÿ����\)��Cs�׿(����
��
=�qC|\)                                    Bx�O �  �          @�Q�G���(�������Cmc׿G�������G���Cx+�                                    Bx�O>  �          @����u�Tz���z�W
C\�)�u�������RB�Cn�3                                    Bx�O�  �          @�33�p�׿\)���R33CR�׿p�׿������\�=CjxR                                    Bx�O,�  �          @�����H���
��p�p�C6O\���H�J=q��33�)CJ                                      Bx�O;0  �          @�\)��\�0����\){CF�)��\��{���\�w\)CU�                                    Bx�OI�  T          @��׿�G��}p����(�CU&f��G���
=��p�L�Cc�                                    Bx�OX|  �          @��\���H�n{��G�CT�{���H�����33=qCdc�                                    Bx�Og"  �          @��ÿ��R�p����
=CTE���R�����G�ffCc��                                    Bx�Ou�  �          @��ÿ�׿#�
��(���CF�׿�׿�=q���(�CW5�                                    Bx�O�n  �          @��׿�
=�����W
CCaH��
=���H����qCT)                                    Bx�O�  �          @�ff��þ�����\)�HC<���ÿ}p���z��|\)CL�{                                    Bx�O��  �          @���Q������=qCC#׿�Q쿗
=������CS\)                                    Bx�O�`  �          @�G��ٙ��(����R�HCG�ٙ�������\33CY.                                    Bx�O�  �          @�\)�������ff��CF�\����������\��CY�                                    Bx�Oͬ  T          @��H�Tz�.{��(�¡8RC?�R�Tz�^�R���33Cbk�                                    Bx�O�R  �          @�\)��  �����RW
CE�\��  ��Q���33�)CZW
                                    Bx�O��  �          @�p���zὣ�
��
=��C7&f��z�G����\)CQ�                                    Bx�O��  �          @��
�Q녾#�
��G�¢\C?+��Q녿^�R��
=.CbǮ                                    Bx�PD  �          @�녿���Ǯ���RCH�������=q���\
=Cb�                                    Bx�P�  �          @��ÿ333�k���£�)CFB��333�k�����\Ch�q                                    Bx�P%�  �          @���\)��G����
¤G�CZ5ÿ\)��{����33Csk�                                    Bx�P46  �          @�z�+�=����R¥
=C)�\�+��\)��{ ��C[�)                                    Bx�PB�  �          @��H��(���=q����¨CT���(��p����=q�fCu�H                                    Bx�PQ�  �          @���0�׾�����¤�HC@5ÿ0�׿W
=��G���Cf�{                                    Bx�P`(  �          @��\��\)>.{���H�\C,�{��\)���H��=q�CGz�                                    Bx�Pn�  �          @��R�   �����z�¤�C_p��   ��������ffCv{                                    Bx�P}t  �          @�{>��R��(���  ��C���>��R��������{C���                                    Bx�P�  �          @��k��5��=q¤  C{�{�k���������C�4{                                    Bx�P��  �          @�  �(�?\)��(�¢� C	�(������¦��C?{                                    Bx�P�f  �          @�Q�z�H?�p������C ��z�H?\)��338RC#�                                    Bx�P�  �          @�Q쾏\)�z���{§  Ct���\)������\aHC��f                                    Bx�PƲ  �          @�Q�>�z��������fC���>�z��9�����\�k�C��
                                    Bx�P�X  
�          @���?�  � ����  �mC�/\?�  �E�����S��C��                                    Bx�P��  �          @\?333��33����C�'�?333�%������~�C��                                     Bx�P�  �          @���>W
=��ff��z�C��H>W
=��\��Q��C�u�                                    Bx�QJ  �          @˅���
��ff����3C�����
��\����C�                                    Bx�Q�  �          @�33��  �����Ǯz�C�����  ������fC�q                                    Bx�Q�  �          @�G����
�Tz���\)¥p�C�zἣ�
�˅�ʏ\�qC���                                    Bx�Q-<  �          @�Q콣�
�s33��£.C������
�ٙ��ȣ���C��                                    Bx�Q;�  �          @��
>��
�\)��=q¨Q�C�� >��
��ff��ff��C��                                    Bx�QJ�  �          @�33>�=q�(����¨aHC�>�=q��\)��{z�C���                                    Bx�QY.  �          @�  ?(������¦Q�C�H�?(����H���HǮC�b�                                    Bx�Qg�  �          @�\)?5�����¤W
C�
=?5��G����8RC���                                    Bx�Qvz  �          @���?(��+���
=¤{C�(�?(���z���33�)C���                                    Bx�Q�   �          @��>�=q������� aHC���>�=q����z�� C�&f                                    Bx�Q��  �          @׮>��ÿ�����33�C�q>�����
�����C��q                                    Bx�Q�l  �          @׮?�Ϳ�=q�ҏ\L�C�Q�?����
��z�
=C��f                                    Bx�Q�  �          @�  >�녿�
=�ҏ\� C��>���
=q��(���C�k�                                    Bx�Q��  �          @�\)>�׿�(��љ�ffC��R>���(����H
=C��                                    Bx�Q�^  �          @�
=>��
��=q�У�(�C���>��
��\�ə���C���                                    Bx�Q�  �          @�{?\)������  (�C�|)?\)�	���ə�#�C�:�                                    Bx�Q�  �          @��?�����
��ffp�C���?���{�Ǯ�=C���                                    Bx�Q�P  �          @��>��Ϳ��
�θR� C�@ >����{��Q�W
C�                                    Bx�R�  �          @��
>�\)��=q��C�3>�\)�����
=��C��\                                    Bx�R�  �          @Ӆ>�  �����p��qC���>�  �{��
=�{C�/\                                    Bx�R&B  �          @��
>�=q�\����C��>�=q����Ǯ�C���                                    Bx�R4�  T          @У�?
=q����˅33C�T{?
=q������{�qC���                                    Bx�RC�  �          @�  >������
��C�˅>���=q�ƸR#�C��                                    Bx�RR4  �          @θR>L�Ϳ�Q����H\)C���>L�Ϳ������=C�%                                    Bx�R`�  �          @��=L�Ϳ������ B�C�h�=L�Ϳ޸R��p�ffC���                                    Bx�Ro�  �          @�ff�B�\������HG�C�*=�B�\�����{��C���                                    Bx�R~&  �          @�
=��Q쿳33����C�B���Q���\��z�=qC��                                    Bx�R��  �          @�����\)����33C��=���   �Å��C�O\                                    Bx�R�r  �          @�{�#�
������G�\C�}q�#�
������(���C��)                                    Bx�R�  �          @�ff<#�
�����G�C�/\<#�
� �����
�{C�                                      Bx�R��  �          @�ff=�Q쿯\)��G�\)C���=�Q���R��(�C�AH                                    Bx�R�d  �          @Ϯ����Q���=q\)C�o\���33��z�{C���                                    Bx�R�
  �          @θR>�p����\��33 ��C�޸>�p������
=�)C�O\                                    Bx�R�  �          @θR>\)��G���(�¡�HC���>\)��\)��  ��C�z�                                    Bx�R�V  T          @�
=>\)��\)���
�HC�xR>\)��p��Ǯ�HC�AH                                    Bx�S�  �          @�  >�����G���p�¡ffC�P�>�����\)��G��RC�Ff                                    Bx�S�  �          @�
=?
=q�z�H���
 �C��f?
=q��=q��  #�C�}q                                    Bx�SH  �          @���?aG���  ��\)C�N?aG��������C�Q�                                    Bx�S-�  �          @��?}p���ff��ff8RC�o\?}p��Q���  �=C�\)                                    Bx�S<�  �          @��
?���\)��
=�C���?���B�\��
=�n�HC�Ff                                    Bx�SK:  �          @��?z�H������
Q�C��?z�H�@  ����vC�f                                    Bx�SY�  �          @�(�?xQ������33\C��?xQ��<(���33�xp�C�"�                                    Bx�Sh�  �          @��?�z��/\)��ff�}��C���?�z��QG����jffC��                                    Bx�Sw,  �          @�?����,(�������C���?����N{��  �m�
C�.                                    Bx�S��  �          @��
?5�Ǯ��z�  C�B�?5����\)u�C�AH                                    Bx�S�x  �          @Ӆ?W
=�
=��ff�qC���?W
=�)�����(�C���                                    Bx�S�  T          @��
?Tz��(����Q��C��)?Tz��J=q��Q��q��C�^�                                    Bx�S��  �          @�p�?c�
��\��{�3C���?c�
�5���
=�~z�C��R                                    Bx�S�j  �          @�p�?h�ÿ�\)�ʏ\�\C���?h���=q��z�k�C�W
                                    Bx�S�  �          @��?u��
=�ə�#�C�C�?u�{�Å�C��                                    Bx�Sݶ  �          @�{?��\����ʏ\�=C�1�?��\��H�����3C�l�                                    Bx�S�\  �          @�{?�  ����Q�C��H?�  �(�����\C�]q                                    Bx�S�  �          @�?xQ��p���\)��C���?xQ��.�R�����3C��H                                    Bx�T	�  �          @�?��\����z�� C�n?��\�;���p��y�C��                                     Bx�TN  T          @�ff?����!���33#�C��?����AG����
�t�RC�H�                                    Bx�T&�  �          @�
=?���#33�Å#�C���?���C33��(��tC��3                                    Bx�T5�  �          @�ff?�33�z����=C�9�?�33�3�
��
=�|
=C�%                                    Bx�TD@  �          @�ff?�Q�������
C��?�Q��;������w�C�{                                    Bx�TR�  �          @�{?�
=�"�\�\�=C�o\?�
=�AG�����t(�C��f                                    Bx�Ta�  �          @ָR?�Q��$z��\��C�h�?�Q��C33����s(�C���                                    Bx�Tp2  �          @ָR?��
�'
=��G��\C�
?��
�E���=q�p��C�P�                                    Bx�T~�  �          @�ff?�{�,����\)�|�C�aH?�{�J=q��  �lffC���                                    Bx�T�~  �          @�
=?��.�R��
=�z�HC��\?��L(�����j��C��
                                    Bx�T�$  �          @�  ?�
=�*�H�����}G�C��?�
=�G������mz�C�N                                    Bx�T��  �          @�Q�?�=q�!G���(�\C��?�=q�>�R��p��t\)C�H                                    Bx�T�p  �          @أ�?�(��{���C�33?�(��;���
=�x{C�Y�                                    Bx�T�  �          @�Q�?�{� �����
��C�,�?�{�>{����tz�C�H�                                    Bx�Tּ  �          @أ�?�  �'
=����~ffC��?�  �C33��33�oQ�C�3                                    Bx�T�b  �          @���?�Q��&ff���H
=C�z�?�Q��B�\��(��q  C���                                    Bx�T�  �          @���?\�!���33\)C�}q?\�>{�����r
=C���                                    Bx�U�  �          @ٙ�?Ǯ�'
=�\�}��C�j=?Ǯ�B�\��(��o=qC���                                    Bx�UT  �          @�=q?���$z�����C�~�?���@����\)�u  C�                                    Bx�U�  �          @��
?��H�'
=��W
C��3?��H�C33��\)�r�C�Ǯ                                    Bx�U.�  �          @�(�?�=q�'
=��\)�3C��f?�=q�B�\�����t�RC���                                    Bx�U=F  �          @�z�?�33�%��Ǯ��C�7
?�33�@����G��t�
C�p�                                    Bx�UK�  �          @���?�  �#�
��G�=qC��
?�  �?\)���H�w��C�L�                                    Bx�UZ�  �          @�(�?�  �'��Ǯ(�C���?�  �B�\�����u�HC�'�                                    Bx�Ui8  T          @�p�?�ff�2�\��ff�~=qC�p�?�ff�Mp�����o�HC��q                                    Bx�Uw�  �          @�?���8Q�����{ffC�4{?���R�\���R�m�C��{                                    Bx�U��  �          @�{?���2�\��
=�~��C�k�?���L�������p��C��)                                    Bx�U�*  �          @�?�G��3�
��ff�~=qC��?�G��N{��  �p(�C���                                    Bx�U��  �          @���?�ff�.�R�ƸR�C���?�ff�HQ���Q��r\)C�:�                                    Bx�U�v  �          @���?�\)�.{��{��C�ff?�\)�G���  �q�\C���                                    Bx�U�  �          @��?���,����ff�C��f?���E��Q��r\)C�)                                    Bx�U��  �          @���?���(Q���\)  C���?���AG������t��C�W
                                    Bx�U�h  �          @�?���%��ə�ǮC�E?���>�R���
�x\)C��{                                    Bx�U�  �          @޸R?�33�)�������fC��\?�33�C33��33�t�
C�`                                     Bx�U��  �          @߮?���%���33k�C�  ?���>{��p��x  C���                                    Bx�V
Z  �          @�  ?�G��$z����H��C�.?�G��=p���p��v�
C�}q                                    Bx�V   �          @�Q�?����1G���\)�{33C��
?����I�������n�
C�p�                                    Bx�V'�  �          @�?�{�:=q��z��y��C���?�{�Q����R�l�HC�Ff                                    Bx�V6L  T          @�ff?���<����z��x��C�E?���S�
��ff�l�C�                                    Bx�VD�  T          @�\)?��J�H��33�s��C�  ?��a������g{C�+�                                    Bx�VS�  �          @�Q�?�{�G���p��w  C��f?�{�^�R��
=�j(�C��{                                    Bx�Vb>  �          @��?����@  ��  �{�RC�
=?����W
=��=q�o
=C��                                    Bx�Vp�  �          @�Q�?���?\)��  �|z�C���?���U��=q�o�HC��\                                    Bx�V�  T          @���?�33�@  ��  �{33C�t{?�33�W
=��=q�n�
C�o\                                    Bx�V�0  �          @�G�?�
=�AG��Ǯ�z{C���?�
=�W�����m�HC���                                    Bx�V��  �          @�G�?����@  �ȣ��|{C���?����Vff���H�o�
C�޸                                    Bx�V�|  �          @�Q�?�
=�1G��ʏ\33C��f?�
=�G���p��v�\C�Y�                                    Bx�V�"  T          @�
=?��
�9����Q��Q�C���?��
�O\)���H�sQ�C���                                    Bx�V��  �          @�
=?fff�4z������C���?fff�J=q�����w��C��3                                    Bx�V�n  �          @߮?G��,(�����C��?G��A���  �}�C�9�                                    Bx�V�  �          @��?p���-p�������C���?p���C33�Ǯ�{��C���                                    Bx�V��  �          @�\)?�  �,���˅8RC�+�?�  �B�\�ƸR�z�C�!H                                    Bx�W`  �          @߮?��\�!G���{B�C���?��\�6ff��G���C�˅                                    Bx�W  �          @�
=?J=q�L(���z��wp�C��{?J=q�`����
=�k�
C�Y�                                    Bx�W �  �          @�Q�?u�Q��У�{C�H?u�-p���z��\C�                                    Bx�W/R  �          @�ff?.{��H��
=�\C��
?.{�0  ���H�fC���                                    Bx�W=�  
�          @�
=?Q������\)aHC�}q?Q��.{��33�)C�l�                                    Bx�WL�  �          @߮?h���)����z���C�u�?h���>{��  �~ffC���                                    Bx�W[D  �          @���?Y���,����G���C��?Y���@  �����|Q�C��=                                    Bx�Wi�  �          @�33?����/\)��p���C���?����B�\�����wp�C��R                                    Bx�Wx�  �          @׮?���E��z��sC�u�?���XQ���\)�i33C��R                                    Bx�W�6  �          @�?�  �P  ��  �m��C���?�  �aG����H�c=qC���                                    Bx�W��  �          @�
=?xQ�����{�C�1�?xQ��*=q�\C��                                    Bx�W��  �          @�
=?h���I����33�r�
C�3?h���[���ff�hp�C�w
                                    Bx�W�(  �          @�Q�?��\�A���ff�v�C�G�?��\�S�
����l��C��\                                    Bx�W��  �          @�=q?�  �'
=�ƸR�\C�s3?�  �9�����H�}33C��H                                    Bx�W�t  �          @�(�@p��@  ���^33C�+�@p��O\)��G��U�C�q                                    Bx�W�  �          @�
=@��L(���  �X��C�AH@��[���33�P  C�Z�                                    Bx�W��  �          @��@��L(���z��V
=C���@��Z=q��  �MffC���                                    Bx�W�f  
�          @�{@(��Vff�����N\)C���@(��dz���z��E�
C�Ǯ                                    Bx�X  �          @�  @ ���C33��z��Q�C���@ ���Q���  �I�
C��                                     Bx�X�  �          @�=q@(��J=q��ff�Q�HC�Ф@(��XQ������I�C���                                    Bx�X(X  �          @�Q�@*�H�:=q��p��R\)C�Ff@*�H�HQ���G��K
=C�@                                     Bx�X6�  �          @љ�@#33�C33����Q�C��
@#33�QG������J  C��                                    Bx�XE�  �          @��@�
�G���ff�Y
=C��\@�
�U����P��C���                                    Bx�XTJ  �          @�@{�;���Q��\33C���@{�I����z��T�\C���                                    Bx�Xb�  �          @�=q@�\�>�R��z��\z�C���@�\�L����Q��T��C�Ф                                    Bx�Xq�  �          @Ӆ@
=�@����(��Z\)C��@
=�N�R��Q��R��C��                                    Bx�X�<  �          @�p�@z��Dz���ff�[(�C��@z��Q���=q�SC��H                                    Bx�X��  �          @��@{�I������T  C��@{�W
=���L��C�"�                                    Bx�X��  �          @��H@���B�\��p��^=qC���@���P  �����V��C��                                    Bx�X�.  �          @��?����C�
��ff�az�C�B�?����P�����\�Y��C�k�                                    Bx�X��  T          @У�?�ff�1G����s(�C���?�ff�?\)��=q�kz�C���                                    Bx�X�z  �          @��H?�Q��7���
=�rC�P�?�Q��E����k  C��H                                    Bx�X�   �          @�\)@z��C�
����ap�C��@z��QG���{�ZG�C�1�                                    Bx�X��  �          @�33@'
=�P�����QC�N@'
=�]p�����K{C�|)                                    Bx�X�l  �          @�ff@{�QG�����WQ�C���@{�^{����P��C���                                    Bx�Y  �          @�{@���R�\��33�W{C�Y�@���_\)��
=�P\)C���                                    Bx�Y�  �          @߮@�H�U������W�C�  @�H�a������P�HC�7
                                    Bx�Y!^  �          @�  @#33�S33���
�U��C��q@#33�_\)��Q��O\)C��                                    Bx�Y0  �          @߮@���W���(��V��C���@���dz���Q��PG�C���                                    Bx�Y>�  �          @�Q�@z��]p���(��U�HC��\@z��j=q��Q��OG�C�:�                                    Bx�YMP  �          @���@�
�\(����W�\C��@�
�hQ�����Q
=C�5�                                    Bx�Y[�  T          @�G�@�H�c33���\�Q�C�%@�H�n�R��ff�K�C�y�                                    Bx�Yj�  �          @ᙚ@%�b�\�����N
=C�
@%�n{�����G�
C�j=                                    Bx�YyB  �          @ᙚ@�R�k���\)�L{C���@�R�vff��33�E��C�^�                                    Bx�Y��  �          @޸R@���`  ��  �P��C��H@���j�H��(��J�C�ٚ                                    Bx�Y��  �          @߮@#33�`  ����N�
C��@#33�k����
�H�HC�aH                                    Bx�Y�4  �          @�  @)���\����\)�N(�C��q@)���hQ����
�H\)C�3                                    Bx�Y��  �          @�Q�@-p��`  ���K=qC��R@-p��j�H��=q�E�\C�4{                                    Bx�Y  �          @�{@:=q�n�R���\�<=qC���@:=q�xQ����R�6��C�h�                                    Bx�Y�&  �          @���@G
=�k���{�6��C�R@G
=�u���=q�1p�C���                                    Bx�Y��  �          @߮@B�\�j�H����<�C��{@B�\�tz���  �6��C�B�                                    Bx�Y�r  �          @�Q�@<(��e��  �B{C���@<(��o\)��z��<C�
                                    Bx�Y�  �          @�Q�@A��`����Q��BG�C�j=@A��j=q�����=�C�Ф                                    Bx�Z�  �          @ᙚ@U�c�
���H�8�
C��R@U�l����\)�3�C�                                    Bx�Zd  �          @�@S�
�aG����R�<z�C���@S�
�j�H��33�7��C�                                    Bx�Z)
  �          @�@HQ��`  ���H�B��C��@HQ��i������=C�P�                                    Bx�Z7�  �          @�z�@<���^{����I=qC�1�@<���g���z��DQ�C��
                                    Bx�ZFV  �          @�=q@5��S33��=q�O�C�O\@5��\����
=�K�C���                                    Bx�ZT�  �          @�@8Q��XQ���G��M�C�8R@8Q��a���ff�HQ�C��)                                    Bx�Zc�  �          @�z�@;��[������K33C�H�@;��e���{�Fz�C���                                    Bx�ZrH  �          @��
@5�Z�H����Mp�C��3@5�dz���
=�HC�=q                                    Bx�Z��  �          @�33@9���Z�H��  �K\)C�&f@9���c�
����F�RC��3                                    Bx�Z��  �          @�33@Dz��`  ��33�C��C��=@Dz��hQ���Q��?z�C��                                    Bx�Z�:  �          @�(�@Fff�`  ��z��D�C�˅@Fff�hQ������?�RC�AH                                    Bx�Z��  �          @�(�@G��^�R��(��D
=C���@G��g
=��G��?�RC�k�                                    Bx�Z��  �          @�p�@S�
�Tz���p��D�\C�l�@S�
�]p����\�@z�C��)                                    Bx�Z�,  �          @���@Tz��U���
�B�C�l�@Tz��^{��G��>�C��H                                    Bx�Z��  �          @�(�@U�Q���(��C�
C��)@U�Z=q�����?�C�.                                    Bx�Z�x  �          @�@Q��U��33�C(�C�8R@Q��^{��Q��?=qC���                                    Bx�Z�  �          @��H@J�H�Q����G�RC�  @J�H�Z=q��33�C��C�t{                                    Bx�[�  �          @�33@HQ��N{��  �K{C�\@HQ��Vff��p��G33C���                                    Bx�[j  �          @�z�@O\)�Mp���  �Iz�C�� @O\)�U����E�RC�3                                    Bx�["  �          @�{@L(��L�����H�L(�C�o\@L(��U���Q��Hz�C��                                    Bx�[0�  �          @�R@P���S33�����G�RC�U�@P���Z�H��{�D
=C�Ф                                    Bx�[?\  �          @�
=@H���U����\�J�C��\@H���\����Q��G  C�+�                                    Bx�[N  �          @�
=@J�H�XQ������HQ�C��3@J�H�`  ���R�D��C�{                                    Bx�[\�  �          @�\)@N�R�[���\)�EQ�C��@N�R�c33����A�RC�+�                                    Bx�[kN  �          @�
=@L(��c�
����B=qC���@L(��j�H���\�>��C�y�                                    Bx�[y�  �          @�
=@J�H�b�\���CQ�C���@J�H�i������?C�w
                                    Bx�[��  �          @���@L(��^{�����F�C�U�@L(��e���\)�C33C��                                     Bx�[�@  �          @�@Vff�g���
=�?�RC�^�@Vff�n�R��z��<G�C��                                    Bx�[��  �          @�@Q��g���  �A��C�  @Q��n�R���>33C��3                                    Bx�[��  �          @陚@N{�e�����C  C��@N{�k���p��?��C��
                                    Bx�[�2  �          @�Q�@P���k����H�=G�C��@P���r�\��Q��9�C�aH                                    Bx�[��  �          @�G�@S33�k����H�<�
C��3@S33�q������9�\C��\                                    Bx�[�~  �          @�G�@Mp��j�H����?��C��{@Mp��qG����H�<�C�1�                                    Bx�[�$  T          @�G�@Dz��n�R���R�A�C���@Dz��u���z��>��C�U�                                    Bx�[��  �          @�\@E�qG���\)�A\)C��@E�w�����>{C�G�                                    Bx�\p  �          @�R@7��qG���p��Cz�C��@7��w���33�@(�C�K�                                    Bx�\  �          @��@8Q��xQ����A=qC�B�@8Q��~�R����=�C���                                    Bx�\)�  �          @�(�@6ff�xQ���ff�<�
C�(�@6ff�~{��(��9�\C��R                                    Bx�\8b  �          @��H@l�����������'\)C�Ff@l�����
���\�$ffC���                                    Bx�\G  �          @��H@l��������z��'33C�H�@l�������=q�$G�C���                                    Bx�\U�  T          @��H@`  ��z���{�)�\C�q@`  ��
=���
�&��C���                                    Bx�\dT  �          @�z�@U���ff���\�-��C�1�@U���G���  �*��C��                                    Bx�\r�  �          @�(�@W���{����-33C�h�@W���������*=qC�#�                                    Bx�\��  �          @�=q@J�H�s�
��(��=Q�C��@J�H�x������:z�C��
                                    Bx�\�F  �          @�(�@H�������ff�3��C��f@H����  ��(��0�C�AH                                    Bx�\��  �          @��
@C�
��������1��C�˅@C�
������H�.�C��=                                    Bx�\��  �          @���@Fff�~{��  �8�
C��@Fff������{�6
=C���                                    Bx�\�8  �          @�{@I����Q�����2��C�
@I�����H����0(�C��3                                    Bx�\��  �          @�R@I���������\�3p�C�@I����33�����0�C�˅                                    Bx�\ل  �          @��@7
=�z=q�����=�HC��@7
=�~�R��
=�;�C�Ф                                    Bx�\�*  �          @�{@^{�y����p��-
=C��{@^{�~{����*�C���                                    Bx�\��  T          @陚@�����G������ffC�u�@����������  C�9�                                    Bx�]v  �          @��H@q��l(���ff�&��C���@q��p�������$\)C��R                                    Bx�]  �          @���@}p��\)�����C�e@}p���������33C�,�                                    Bx�]"�  �          @�ff@�  �~�R����  C��@�  ��G�����RC�T{                                    Bx�]1h  �          @��@�  �w
=�����p�C��@�  �z�H���R�=qC��\                                    Bx�]@  �          @�
=@��i�������HC�n@��n{������C�0�                                    Bx�]N�  �          @��@�  �n�R��=q�C�W
@�  �r�\������RC�)                                    Bx�]]Z  T          @�Q�@��\�u�����
C�7
@��\�y��������C�H                                    Bx�]l   �          @���@�=q����������C��@�=q��=q��
=�z�C�\)                                    Bx�]z�  �          @陚@�ff�������
�	C���@�ff��33��=q�C���                                    Bx�]�L  �          @�
=@���\)��ff�ffC���@����G������ffC�j=                                    Bx�]��  T          @�Q�@����x����(���\C�޸@����|�����\���C���                                    Bx�]��  T          @�G�@�p��n�R����
C���@�p��r�\��z��  C��{                                    Bx�]�>  �          @�(�@�z��u������=qC�t{@�z��x����\)�ffC�AH                                    Bx�]��  �          @��@�p��{������  C�'�@�p��\)��  ��C��R                                    Bx�]Ҋ  �          @��H@���x������z�C�w
@���|(���{���C�H�                                    Bx�]�0  �          @�33@w
=����������C�˅@w
=��������C��q                                    Bx�]��  �          @��H@x����  ����
=C�)@x�����������C��                                    Bx�]�|  �          @�@�p��x����  ���C�}q@�p��|(���ff�  C�O\                                    Bx�^"  �          @�=q@�(��y����ff���C�L�@�(��|������(�C�!H                                    Bx�^�  �          @���@�(��x���������C�XR@�(��|(�����
=C�,�                                    Bx�^*n  �          @�Q�@�z��xQ��vff�Q�C�AH@�z��z�H�s�
���C��                                    Bx�^9  �          @��@���q��Z=q��(�C�@���tz��W���
=C��                                    Bx�^G�  �          @ᙚ@����E�
=��=qC�4{@����G
=�����C�)                                    Bx�^V`  �          @��H@Å�?\)���R����C���@Å�@�׿��H����C��H                                    Bx�^e  �          @��H@�Q��`���
�H��z�C�L�@�Q��b�\�Q����C�7
                                    Bx�^s�  �          @��
@����g
=�������C���@����hQ��ff��ffC���                                    Bx�^�R  �          @�(�@�Q��`����\��ffC�Q�@�Q��a�������C�:�                                    Bx�^��  �          @��
@�Q��]p��ff���C��@�Q��^�R��
��p�C�o\                                    Bx�^��  �          @�(�@�  �_\)�
=���\C�]q@�  �`���z���(�C�G�                                    Bx�^�D  �          @��@��\�a��.�R��  C��q@��\�c33�,�����C�                                    Bx�^��  �          @��@�\)�n{�
=��33C�~�@�\)�o\)�z����RC�l�                                    Bx�^ː  �          @�z�@�z��W��{���HC�)@�z��X��������C��                                    Bx�^�6  �          @�z�@���u��)����(�C�XR@���w
=�'
=����C�AH                                    Bx�^��  �          @�(�@��
�w
=�(����=qC�"�@��
�x���'
=���C��                                    Bx�^��  �          @�\@�G��e��&ff��=qC���@�G��fff�$z����
C�y�                                    Bx�_(  �          @�\@��H�]p��'����
C�  @��H�^�R�%���C��                                    Bx�_�  �          @��H@��R�dz��1����\C�j=@��R�e�/\)��=qC�Q�                                    Bx�_#t  �          @�=q@�{�mp��C33�̣�C�<)@�{�n�R�AG���(�C�!H                                    Bx�_2  T          @�=q@�  �g
=�E���ffC���@�  �hQ��B�\��  C���                                    Bx�_@�  �          @�33@�
=�aG��5����C��H@�
=�c33�3�
��Q�C���                                    Bx�_Of  �          @�@�{�R�\�-p���
=C���@�{�S�
�+�����C���                                    Bx�_^  �          @�(�@�\)�P���.{��G�C�*=@�\)�R�\�,(���33C�3                                    Bx�_l�  �          @��
@����R�\�#33��{C�(�@����S�
�!G���{C�3                                    Bx�_{X  �          @�z�@���\���6ff��ffC�3@���^�R�4z���(�C���                                    Bx�_��  �          @���@�p��`  �E���  C��3@�p��a��C33��C�xR                                    Bx�_��  �          @���@����a��E����
C�e@����c�
�B�\�ə�C�K�                                    Bx�_�J  �          @���@�G��`  �6ff��  C��@�G��aG��4z����
C��=                                    Bx�_��  �          @�p�@��
�Z=q�6ff��C�Z�@��
�\(��4z�����C�B�                                    Bx�_Ė  �          @���@�G��^{�8����z�C��
@�G��`  �7
=��Q�C��                                     Bx�_�<  �          @�p�@��R�j�H�33��{C���@��R�l(��G����
C���                                    Bx�_��  �          @�(�@���fff��H����C���@���g�������\C���                                    Bx�_��  T          @�@���g
=�%���{C���@���hQ��#33���C�l�                                    Bx�_�.  �          @�(�@�G��hQ��(����C�e@�G��i���&ff���C�O\                                    Bx�`�  �          @��@�G��a��4z�����C���@�G��c�
�2�\��p�C���                                    Bx�`z  �          @��@��R�o\)����=qC�aH@��R�p�����{C�P�                                    Bx�`+   �          @���@����l(������RC�l�@����mp�����\C�Z�                                    Bx�`9�  �          @��@��H�n{��R��{C�*=@��H�o\)������
C�
                                    Bx�`Hl  �          @�z�@��R�vff�\)��C�j=@��R�w��p���p�C�W
                                    Bx�`W  �          @��H@��u������{C�aH@��w
=��H���
C�N                                    Bx�`e�  �          @�=q@�Q��tz��p���  C���@�Q��u����C���                                    Bx�`t^  �          @�G�@��R�x���Q����C�E@��R�z=q�ff���HC�5�                                    Bx�`�  �          @߮@�p��s33�  ��{C�z�@�p��tz��{���
C�h�                                    Bx�`��  �          @��@�  �u�33��Q�C���@�  �vff�G���{C�|)                                    Bx�`�P  �          @�Q�@�(��y����R��Q�C�@�(��z�H�(���  C��{                                    Bx�`��  �          @�=q@��R�r�\�����ffC���@��R�s�
�
=��(�C���                                    Bx�`��  �          @�G�@��H�vff�{���\C�q@��H�w��(���=qC�
=                                    Bx�`�B  �          @�\)@�z��xQ��
=���HC�!H@�z��y�������\C��                                    Bx�`��  �          @ۅ@����qG���R��(�C�33@����r�\������
C�!H                                    Bx�`�  �          @�@���r�\����{C�~�@���s�
�33��C�n                                    Bx�`�4  �          @޸R@���u�����C��@���w
=�33���\C���                                    Bx�a�  �          @�
=@�{�|���(����C�aH@�{�~{������C�N                                    Bx�a�  
*          @�
=@��\�u��ff���C�#�@��\�vff�z����C��                                    Bx�a$&  "          @߮@�=q�w
=�Q���33C��@�=q�xQ�����HC���                                    Bx�a2�  
�          @�  @�  �o\)����
=C��@�  �p���	�����RC�Ф                                    Bx�aAr  
�          @�{@�
=�n�R�ff����C��
@�
=�p  ��
��Q�C��f                                    Bx�aP  "          @�
=@�ff�mp������33C��@�ff�n�R�{���HC��\                                    Bx�a^�  
Z          @�@�(��n{��\��ffC��f@�(��o\)�����{C��3                                    Bx�amd  �          @�\)@�(��n{����33C���@�(��o\)�������C��{                                    Bx�a|
  �          @��@���p���Q����C���@���q��ff��G�C���                                    Bx�a��  
�          @�G�@�{�qG�������RC��q@�{�s33�ff��Q�C��=                                    Bx�a�V  �          @�Q�@���l(������ffC�33@���mp��
=��  C�!H                                    Bx�a��  
Z          @��H@��R�|(��  ��  C�
@��R�}p��{���C�                                    Bx�a��  �          @��@��H�x����R��\)C��
@��H�z=q������HC���                                    Bx�a�H  
Z          @�@�p��j�H�����z�C���@�p��l(��
�H��{C�xR                                    Bx�a��  "          @�33@����p  ���33C�7
@����qG���
����C�%                                    Bx�a�  T          @�\@��H�r�\�������C��\@��H�s�
�ff��Q�C��)                                    Bx�a�:  T          @�33@�33�vff��\���
C��)@�33�w��   ��G�C��=                                    Bx�a��  �          @�=q@���j=q�#33���C�!H@���l(��!G�����C��                                    Bx�b�  T          @��H@����n�R�+����C���@����p  �(�����C���                                    Bx�b,  �          @��H@��l���)�����C��R@��n�R�'���\)C��                                     Bx�b+�  
�          @ᙚ@�(��r�\�"�\���HC�p�@�(��s�
�   ��(�C�XR                                    Bx�b:x  �          @��@�
=�k�� ������C�f@�
=�mp��{��(�C��                                    Bx�bI  �          @��@�p��g
=�&ff���
C�'�@�p��h���#�
��33C�\                                    Bx�bW�  
Z          @�=q@��R�j�H�%���G�C��@��R�l���"�\���\C��{                                    Bx�bfj  
�          @��@�G��s�
�����Q�C��)@�G��u�=q����C��f                                    Bx�bu  �          @�(�@����u�
=����C��)@����w
=�z����
C��                                    Bx�b��  
�          @�33@����w
=��R��(�C��@����x���(���\)C�p�                                    Bx�b�\  �          @��
@��R�y���Q���{C�=q@��R�z�H����33C�'�                                    Bx�b�  
�          @�z�@�=q�}p��'���Q�C��=@�=q�\)�$z���G�C���                                    Bx�b��  "          @�z�@����u�)����ffC�J=@����w��'
=��p�C�0�                                    Bx�b�N  �          @�{@�G��{��z����\C�S3@�G��}p��G�����C�<)                                    Bx�b��  
�          @�p�@�����H�������C��f@���������C���                                    Bx�bۚ  
�          @�R@��
������
���\C�/\@��
����� ����p�C��                                    Bx�b�@  
�          @�@���|(�����
C�t{@���~{�33���RC�`                                     Bx�b��  
�          @�{@��s�
�\)��
=C��@��u��(���  C���                                    Bx�c�  T          @���@���|(��33���C�xR@���}p��   ���C�c�                                    Bx�c2  "          @�33@�=q�}p�����v�\C�G�@�=q�\)���p(�C�4{                                    Bx�c$�  "          @�G�@�����G�� ����z�C��
@�����녿��H���C���                                    Bx�c3~  �          @�\)@���xQ���
����C�*=@���z=q� ����=qC�{                                    Bx�cB$  T          @��@�{�����������\C���@�{��=q��33�z=qC��
                                    Bx�cP�  
�          @��@�
=�~{��\��Q�C�@�
=�\)���R����C��                                    Bx�c_p  
�          @��@��\)���H��{C���@����׿�z��}�C��                                     Bx�cn  
(          @�
=@�����p���\)�x��C���@�����ff���qG�C��R                                    Bx�c|�  �          @�\)@�ff��\)������  C�q�@�ff��  ��33�|Q�C�\)                                    Bx�c�b  �          @�=q@�
=���
������C��)@�
=������
��
=C��H                                    Bx�c�  "          @޸R@�p���ff��\��(�C�q�@�p���\)��p���=qC�Z�                                    Bx�c��  T          @�\)@�33��G���
���C��R@�33����   ���C��H                                    Bx�c�T  
�          @޸R@�������Q���Q�C��3@������H�z���(�C���                                    Bx�c��  �          @�G�@�33���\�
�H���
C���@�33����
=���C��H                                    Bx�cԠ  
�          @���@�����Q��(���33C�8R@�����G��Q���
=C��                                    Bx�c�F  
�          @��H@�
=��
=����ffC���@�
=��  �{��(�C�o\                                    Bx�c��  T          @��H@�p����R������
C�g�@�p���  ������C�J=                                    Bx�d �  
�          @�\)@�����\�������C���@������(���z�C�l�                                    Bx�d8  
�          @�  @�����Q���
��Q�C��@�����G��  ���
C��{                                    Bx�d�  
�          @�@�(����
�  ���C��@�(������(���p�C���                                    Bx�d,�  "          @�=q@�G����\�=q��C���@�G����
�ff���C��=                                    Bx�d;*  "          @���@��\���\�"�\���\C��=@��\���
�{���C���                                    Bx�dI�  
�          @��@�Q���p��&ff��  C��@�Q����R�!���G�C��)                                    Bx�dXv  S          @�@����R�!G���=qC��@���  �����G�C��                                    Bx�dg  U          @�\@������!���33C��q@�����������{C�}q                                    Bx�du�  
�          @�@�{�����%�����C�'�@�{��{� ������C�                                    Bx�d�h  
�          @��H@��R��(��#33��ffC�G�@��R��p��{��\)C�%                                    Bx�d�  �          @��@�{��(�� ����=qC�8R@�{��p�����
=C��                                    Bx�d��  T          @�@��������%���C���@������H�!G���  C���                                    Bx�d�Z  "          @�@�  ��=q�'����RC��
@�  ���
�"�\��p�C�q�                                    