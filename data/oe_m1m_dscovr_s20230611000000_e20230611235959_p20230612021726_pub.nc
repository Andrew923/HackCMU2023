CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230611000000_e20230611235959_p20230612021726_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-06-12T02:17:26.210Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-06-11T00:00:00.000Z   time_coverage_end         2023-06-11T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�|
@  �          A3\)@E>�p�A,Q�B��q@�(�@E�p�A(��B�\)C�8R                                    Bx�|�  
n          A.�H@%�?�A'�B���A��@%���\)A(  B�\C�E                                    Bx�|'�  T          A*�H@��?�
=A#\)B���B33@�ͿL��A$��B�aHC��
                                    Bx�|62  �          Ap�?���@G�@�{B��Bl��?���>�p�A (�B�#�Atz�                                    Bx�|D�  �          @�?�G�@�33?uAB�z�?�G�@�
=@��A��
B��                                    Bx�|S~  �          @�p�?�=q@�z�?.{@�z�B�B�?�=q@��\?�Q�A�ffB�W
                                    Bx�|b$  
�          @��H?�{@�  ?��A�G�B�\?�{@�p�@=p�B(�B��)                                    Bx�|p�  �          @�z�?�
=@���?��RA�{B��3?�
=@{�@@��B�HB�u�                                    Bx�|p  �          @��
?�ff@�G�@ffA�B��?�ff@���@N�RB�RB��                                    Bx�|�  �          @�(�?��
@�Q�@o\)B �B���?��
@P��@�z�BS�B��                                    Bx�|��  �          @�(�?�@E@��RBkffB���?�?��H@���B�#�B^�H                                    Bx�|�b  �          @ə�?���@I��@��Bh��B�\)?���?�ff@�ffB�8RBl��                                    Bx�|�  T          @�{?�=q@[�@���BYz�B��?�=q@	��@�{B��B}�\                                    Bx�|Ȯ  T          @��?+�@�G�@�  B8Q�B�8R?+�@;�@��HBm��B�B�                                    Bx�|�T  �          @��>���@��H@s�
B+ffB��R>���@E�@��Bap�B�\                                    Bx�|��  �          @�=q<��
@���@w
=B(
=B��<��
@P��@�Q�B^\)B�\)                                    Bx�|��  T          @�(�?�
=@�Q�@��BQG�B�L�?�
=@(Q�@�ffB�(�B�Ǯ                                    Bx�}F  T          @��\?��R@�33@˅BYz�B��?��R@   @���B���BMz�                                    Bx�}�  T          @�{?��H@}p�@˅B_�\B�W
?��H@
=@�B�u�BXp�                                    Bx�} �  �          @��R?��
@g�@љ�BiffB�?��
?��R@�RB�Q�B@��                                    Bx�}/8  �          @��@	��@���@ə�BS{B|��@	��@,(�@��
B�aHBM
=                                    Bx�}=�  �          A�H?�p�@��@�{BO��B��?�p�@G
=@��
B�� Bs�                                    Bx�}L�  T          A
�R@��@�\)@�BT�\B�L�@��@?\)@��HB��RBY�\                                    Bx�}[*  �          A�R@*�H@��R@�\)BY=qBl\)@*�H@*=qA�B���B3ff                                    Bx�}i�  �          A��@$z�@��\@�{BX�HBg33@$z�@�H@�RB�W
B,��                                    Bx�}xv  �          A��@�
@�\)@��
BKz�B��@�
@S�
@��HB{\)Bh��                                    Bx�}�  �          Ap�>�@�
=@��B(  B�z�>�@�  @���B]Q�B��)                                    Bx�}��  �          A��?�33@�=q@˅B2�HB�Q�?�33@�@���Bg(�B�L�                                    Bx�}�h  T          A�?��
@�ff@�Q�B6(�B�W
?��
@�G�@���Biz�B�Ǯ                                    Bx�}�  �          A��?��H@���@�ffBOQ�B�  ?��H@aG�@�\)B��B���                                    Bx�}��  T          A  ?��@�=q@��
BO�HB��?��@Vff@�33B�aHBw��                                    Bx�}�Z  �          Az�@@�=q@��
B:
=B�� @@}p�@�\)Bi��Bm�
                                    Bx�}�   �          A�?ٙ�@��
@�\)BDz�B�Q�?ٙ�@c33@�Bu��B���                                    Bx�}��  �          A=q?�p�@�  @�BE�RB�?�p�@\��@�p�Bv��B}Q�                                    Bx�}�L  �          A{?�Q�@�  @�p�BE�HB���?�Q�@\��@���Bv�HB�                                    Bx�~
�  �          @�{?���@�z�@��
B>�RB�k�?���@j�H@�z�Bp��B��q                                    Bx�~�  �          A\)?�=q@��@�G�BJ=qB�.?�=q@[�@��Bz�\Bw��                                    Bx�~(>  �          A�\?�G�@��R@���B>�
B���?�G�@l��@�\Bo�RB�\                                    Bx�~6�  �          A�?�=q@��
@�\)B4��B�W
?�=q@{�@ٙ�Be�
B�                                    Bx�~E�  T          A33?��R@�ff@�z�B'��B��?��R@���@�33BX�
B��=                                    Bx�~T0  �          A�?�@�z�@���B�RB�?�@��@�(�BQ�RB�                                    Bx�~b�  �          A33?��@ڏ\@�{B$�\B�33?��@�  @�=qBVz�B���                                    Bx�~q|  T          A(  >W
=A
=@θRB
=B��>W
=@У�AffBM=qB�aH                                    Bx�~�"  �          A*=q����A�@��HA��HB�#׾���@�@�
=B3�B���                                    Bx�~��  �          A?�?�p�A
=A
=B2G�B�(�?�p�@�=qA!�BeG�B��R                                    Bx�~�n  �          AFff?Tz�A�A�B3�RB�#�?Tz�@���A'�Bg\)B��\                                    Bx�~�  �          AE��?�  A�A(�B.(�B��?�  @�\)A$z�Ba\)B��                                    Bx�~��  �          AC�?�p�A
=A33B/(�B��R?�p�@ҏ\A"�HBb  B��3                                    Bx�~�`  �          AD��@�A
=A
=B4�B��@�@���A%Be�B�.                                    Bx�~�  �          AD  ?���A  A�HB;�RB��)?���@���A(��BnffB�k�                                    Bx�~�  S          AB�\?�  A�A��B2�\B���?�  @˅A#�Bd��B�.                                    Bx�~�R  �          AG33?���A��Ap�B;�
B�G�?���@��A+\)Bm��B���                                    Bx��  �          AH��?���A�RA�B?�HB��?���@���A.�\BqG�B�\)                                    Bx��  �          AH(�@ff@��A\)BD�B�L�@ff@��A/�Bu�B�
=                                    Bx�!D  �          AH(�@��@�(�A�
BT
=B�@��@��A5�B���B|��                                    Bx�/�  �          AF�R@*=q@ə�A%�BbQ�B�.@*=q@w
=A8  B���B]z�                                    Bx�>�  T          AD  @*�H@ÅA$��Bd33B��R@*�H@l(�A6{B�W
BXQ�                                    Bx�M6  �          AC33@0  @��\A%p�Bh(�B��=@0  @Z�HA5�B���BL                                    Bx�[�  �          AEG�@=p�@�
=A((�Bj  Bzz�@=p�@QG�A8  B�
=B?\)                                    Bx�j�  �          AG
=@7�@��
A)p�BiG�B�R@7�@Y��A9B�
=BGp�                                    Bx�y(  T          AA�@,(�@���A"�\Be��B�\@,(�@b�\A3\)B��qBS�                                    Bx���  
�          AB�H@5�@���A#
=Bcp�B��R@5�@i��A4(�B�ffBQ{                                    Bx��t  "          AI�@333@ə�A(��Bcz�B�{@333@vffA:�HB��BW��                                    Bx��  U          AB�\@   @�
=A!G�Ba�B��@   @xQ�A3
=B�.Bd�                                    Bx���  �          AC33@@��
A#33Bb�B��R@@�Q�A5p�B��\By�H                                    Bx��f  �          AB�H@�@��HA$��Bf\)B�z�@�@n{A5�B���Be�H                                    Bx��  T          A;33@|(�@L��A'33B�8RB(�@|(�?\(�A.ffB�aHAD��                                    Bx�߲  T          A@��@~�R@mp�A+�
B|�B,33@~�R?��A4��B�L�A��
                                    Bx��X  �          A=�@!�@���A
=Bi�B�  @!�@Q�A.ffB�k�BQ\)                                    Bx���  
�          A<  ?\@�  A�\B]Q�B��3?\@���A-G�B��B��H                                    Bx���  
�          A8��?Ǯ@�A=qBb33B��R?Ǯ@~{A,  B��qB�#�                                    Bx��J  T          A'�@*=q@HQ�A(�B�8RBF�\@*=q?��
A\)B�A�z�                                    Bx��(�  �          A9G�@AG�?��A1G�B���A�p�@AG�����A1��B��3C��{                                    Bx��7�  "          AD  @P  ?�G�A<z�B�ǮA�{@P  �\A;�B���C�k�                                    Bx��F<  "          AB{@-p�>���A=�B�  @��@-p��{A9�B�ffC�XR                                    Bx��T�  �          A;�
@C33����A5�B���C��@C33�   A0��B���C�Q�                                    Bx��c�  �          A3\)@N{��=qA)��B���C���@N{�r�\A Q�B~�\C�,�                                    Bx��r.  T          A0��@W
=�Q�A$��B�ǮC��R@W
=��Q�A{Br\)C�&f                                    Bx����  "          A/
=@6ff�Q�A%�B�
=C���@6ff��Q�A=qBx(�C��                                    Bx���z  T          A0��@.{�$z�A&=qB�ffC�L�@.{��ffA�Br=qC�                                    Bx���   �          A5�@Dz��{A+\)B���C��@Dz���p�A   Bv��C�g�                                    Bx����  �          A9G�@J�H��\A/�
B�k�C�l�@J�H����A&{B~�RC�R                                    Bx���l  "          A6�\@A녿��A/
=B��C�l�@A��c33A&�HB���C�5�                                    Bx���  #          A8��@.{��33A1p�B���C�Z�@.{�{�A(  B�
=C�XR                                    Bx��ظ  T          AI�@0���z�AA�B�8RC���@0������A6�HB�\C��f                                    Bx���^  
�          AYG�@*=q�X��AMB��C�3@*=q���A>{Btz�C��q                                    Bx���  T          AZff@��Ϳ���AM��B�=qC�e@������\ADQ�B�G�C��H                                    Bx���  T          A_�
@�녿��AT(�B��3C��@�����\AIG�B��C���                                    Bx��P  #          Aa�@j�H>�G�AZ{B��@���@j�H�Q�AV�RB��C�z�                                    Bx��!�  �          Ab=q@w
=��AY��B�L�C���@w
=�5�AT��B�ffC�޸                                    Bx��0�  �          AaG�@z=q���\AW�B���C��
@z=q�p��AO�
B���C��                                    Bx��?B  T          A`Q�@��׿n{AV=qB��RC�|)@����i��AN�RB�\)C��H                                    Bx��M�  �          A`Q�@xQ�@  AW33B�8RC��@xQ��^�RAPQ�B�Q�C�3                                    Bx��\�  
�          Aap�@��H�.{AW�B���C�O\@��H�Z=qAP��B���C�
                                    Bx��k4  �          A`  @��ÿ!G�AVffB�#�C���@����UAO�
B��C�(�                                    Bx��y�  �          A_�
@�ff�J=qAU�B�u�C���@�ff�^{AN=qB�(�C�33                                    Bx����  �          A`��@��
�
=AV�\B�z�C���@��
�R�\AP(�B��)C��
                                    Bx���&  T          Aa�@��\��AW33B���C�XR@��\�N�RAQ�B��{C��{                                    Bx����  �          A`��@��H��\AV�\B���C�xR@��H�L��AP��B��\C�                                      Bx���r  �          Ab=q@������AX(�B���C�Ff@���FffARffB��C��                                    Bx���  "          Ab�R@�p��
=qAXQ�B�k�C�L�@�p��O\)AR{B�.C��                                    Bx��Ѿ  T          Ac�@�����(�AX��B���C�#�@����HQ�AS
=B�{C���                                    Bx���d  T          Ab�R@���
=AX��B��C���@���Q�ARffB�k�C���                                    Bx���
  T          Aa��@�z��AV=qB�p�C�� @�z��J�HAPQ�B���C��                                    Bx����  
�          Ab{@��ÿk�AX(�B�  C���@����e�AP��B�.C�0�                                    Bx��V  T          Ab�\@��Ϳ=p�AXQ�B�Q�C��{@����Y��AQ��B�u�C�\)                                    Bx���  "          AV{@����G�AK�B�
=C��@���<(�AF=qB���C�.                                    Bx��)�  
�          AH��@q녽���A?33B��fC�>�@q����A;33B��fC��                                    Bx��8H  �          AN�\@^{��33AF�RB���C��@^{�1�AAB���C��                                     Bx��F�  "          A<��@?\)?!G�A5�B��HA?33@?\)����A4Q�B�z�C��                                    Bx��U�  T          A2{@(��@XQ�A$  B��\BO�
@(��?��A+\)B�.A�\)                                    Bx��d:  �          A.�H?ٙ�@QG�A$  B�u�By?ٙ�?��HA+
=B��HB\)                                    Bx��r�  
�          A.�\?xQ�@�z�ABzQ�B�u�?xQ�@;�A&{B��)B�=q                                    Bx����  T          A9����HA
=@�(�B�\B�\���H@��\@���B1�\B�#�                                    Bx���,  �          ARff�j�HAE��@ ��A0Q�B���j�HA:{@�33A�\)B�                                    Bx����  
�          AS\)���\AFff?��\@�B�z����\A?�@X��AnffB�                                    Bx���x  �          AR=q��ffAA녿:�H�Mp�B�k���ffA@z�?�{@�G�B�3                                    Bx���  T          AN�H����A7\)��\�"�HB� ����A;
=�L�;W
=B��                                    Bx����  
Z          AQ����\AB=q���Ϳ޸RB�aH���\A?�
?�
=A�B��)                                    Bx���j  T          AR�\�w�AG�
?��H@˅B�k��w�A?�@s�
A���B�Ǯ                                    Bx���  T          AV=q�y��AK33?�(�@�z�B���y��AB=q@�33A���Bי�                                    Bx����  
�          AW33���AL  ?��@��B�(����AD��@_\)ApQ�B�\)                                    Bx��\  "          AV{�~�RAJ=q?�\)A ��B�  �~�RA@��@�
=A�33B؏\                                    Bx��  �          AV�R�\)AJff@G�A
�HB��\)A@��@��
A�  Bب�                                    Bx��"�  
�          AV�\���AHQ�@  A=qB�k����A>{@��A��HB�G�                                    Bx��1N  �          AV{��AH(�@��A\)B����A>{@�  A�\)B�                                    Bx��?�  
�          AQG����RAC
=@Q�A{B�����RA9G�@��
A�{B���                                    Bx��N�  
�          AS\)��\)ADz�@33A Q�B�  ��\)A:=q@�G�A�
=B��                                    Bx��]@  T          AP������AC33@{A��B؊=����A9G�@�{A�\)B�\)                                    Bx��k�  T          AO\)�}p�ABff@
�HAffB���}p�A8��@�(�A�  B��f                                    Bx��z�  
Z          AQ��z=qADQ�@\)A{B�Q��z=qA:=q@��RA��
B��                                    Bx���2  	�          AR{�s�
AEp�@A$  B�W
�s�
A;\)@�=qA���B�{                                    Bx����  �          AUG��p  AI@
�HA�B�#��p  A?�
@�{A�BָR                                    Bx���~  T          AT���n{AI?�
=AB��)�n{A@��@�ffA��B�L�                                    Bx���$  T          AU���`��AJ=q@\)A�HB�\�`��A@z�@�  A�(�BԔ{                                    Bx����  T          AQ�FffAD��@AG�AV=qB�Q��FffA8��@�ffA�{B�{                                    Bx���p  
�          AQ�p��AAp�@VffAmp�Bօ�p��A4z�@��A�{B��
                                    Bx���  
�          A]����ffARff?k�@s33B�u���ffALQ�@QG�AZ�HB�p�                                    Bx���  
�          A`(����AU��?J=q@N{B֣����AO�
@J�HAQp�B׊=                                    Bx���b  �          A_\)��(�AR�\>.{?333B���(�AN�R@!G�A&�RB�k�                                    Bx��  
�          A`z����AT  �B�\�EB������AR�R?�\)@�(�B�.                                    Bx���  T          A`  ���AR=q�8Q��<(�Bۣ����AP��?У�@�ffB��)                                    Bx��*T  T          Ac�
����AT���G���RBٙ�����AW33>�p�?�G�B�8R                                    Bx��8�  �          Ad  ���HAU���Ǯ�˅B��
���HAS\)?�(�@�\)B�=q                                    Bx��G�  "          A[���=qAK�
@  A�HB�u���=qAB=q@��RA��B�=q                                    Bx��VF  �          AW�
�w�AL  @AffB�Ǯ�w�AC
=@�G�A�z�B�B�                                    Bx��d�  T          AX���s�
AK�@*�HA5��B�W
�s�
AA�@�33A��
B�{                                    Bx��s�  T          AZ{�vffAF{@�\)A���B֔{�vffA733@ʏ\A��B�33                                    Bx���8  "          AYG��h��A:�R@�33A��HB֮�h��A(Q�@�B�B�33                                    Bx����  �          AU���EA1@ָRA�B���EAz�A��B{B��                                    Bx����  
Z          AW33����ADz�L�Ϳ\(�B�z�����AB{?�
=A�B��                                    Bx���*  �          A]����z�AL����\��B�����z�AO\)>B�\?J=qB�W
                                    Bx����  T          A_\)���
AO\)�\)��B�8R���
ARff<��
=��
Bڳ3                                    Bx���v  T          A]���AI�<(��E�B�=q���AN�H�@  �FffB�Q�                                    Bx���  "          A]�����AE���[��eG�B�L�����AL  ���
����B���                                    Bx����  �          A]����A<����z����B�\)���AEp����
ffB�ff                                    Bx���h  �          Aa���A:ff��  ����B�u����AG33�l���s�B�\                                    Bx��  
�          A`����\)A>=q��G���33B� ��\)AI�Mp��S�
B�{                                    Bx���  
�          A_33��p�A?����R���\B���p�AJ{�8Q��>�\B��\                                    Bx��#Z  �          A^ff����AG
=�6ff�=B������AL  �:�H�AG�B��                                    Bx��2   "          A^�\��G�AIG��*�H�0��Bᙚ��G�AM�������B�q                                    Bx��@�  
�          A^{��AF=q�;��C�B�Q���AK33�Tz��\��B�G�                                    Bx��OL  �          A]����{AF�R�N{�V�HB�W
��{ALz῏\)��(�B�.                                    Bx��]�  T          Aap����HAM��$z��(Q�B�.���HAQ녾�녿�Q�B�p�                                    Bx��l�  �          A_33���AK�
�У���
=B�
=���AMp�>�?�p�B�R                                    Bx��{>  T          A@���h��A/\)@@��Ai��Bس3�h��A%G�@�Q�A�{B���                                    Bx����  T          AD(��(�Az�AQ�B,�RB����(�@�=qA��BP��B�p�                                    Bx����  �          AT�Ϳh��@�AC�B�ffB��ÿh��@(Q�AM�B�z�B�                                    Bx���0  
�          Ad  ��@@��A^�\B�G�B�B���?8Q�Ac�B���B��                                    Bx����  
�          Ah�þaG�@A�Ac�
B��HB�ff�aG�?333Ah��B�33B�33                                    Bx���|  �          An=q�#�
@:=qAi��B�u�B��\�#�
?�An{B��fB��=                                    Bx���"  
�          Al�׽�@@  Ag�B��\B�����?&ffAl(�B��HB�{                                    Bx����  T          Al�;��H@z�Aj{B�k�B��
���H��\)AlQ�B���CQxR                                    Bx���n  "          Ak��(��?n{Aj�HB��B�k��(�ÿ�z�AjffB��fCt�R                                    Bx���  T          Ak�
�^�R>���Ak33B��CO\�^�R��AiG�B��Cu�)                                    Bx���  "          Anff��ff�(��Am�B��
Ck���ff�?\)Aip�B��{C���                                    Bx��`  
�          Ak����ÿ��Aj�RB�ǮC}�������Z�HAd��B�C�9�                                    Bx��+  T          Ajff��\)���Ah  B���C��\��\)��(�A`z�B�#�C��                                    Bx��9�  
�          Am>#�
��(�Al  B���C���>#�
�\)Ad��B���C�                                      Bx��HR  "          Am?�R���
Ak�
B�.C�{?�R�r�\AeG�B���C���                                    Bx��V�  T          AhQ�>��Tz�Aa�B�33C��3>���ffAV=qB���C�j=                                    Bx��e�  �          Aq녽��
�S33AlQ�B���C�S3���
����Aap�B�.C��R                                    Bx��tD  "          At�ý��
�@  Ap(�B�k�C�9����
��  Af{B��)C��\                                    Bx����  "          Av�H�\)�G
=Aq��B��3C����\)���
Ag33B�=qC�=q                                    Bx����  T          Azff���
�k�As33B�ǮC�^����
��{Ag�B�\)C��)                                    Bx���6  	�          Av=q�L��?333Ae��B�8RB�  �L�Ϳ�(�Ad��B��C�"�                                    Bx����  
�          Ai?�
=A#33A%�B4��B���?�
=Az�A;�BV�HB�B�                                    Bx����  T          Af�R?��\A.�\A�B"{B�\)?��\A=qA.=qBD=qB���                                    Bx���(  
�          AaG�?��R@��HA/�Bc  B�8R?��R@�(�A>�RB�G�B��                                    Bx����  T          Aa�?B�\@���AT��B��)B�L�?B�\@��A\��B�p�B�u�                                    Bx���t  "          Ab=q�z�@�33A;�B`�B�uÿz�@�G�AL  B�#�B�                                      Bx���  �          AdQ쿘Q�Ap�A'�B=(�B����Q�@�ffA<(�B^��B���                                    Bx���  �          Abff�!G�A33A.�RBJffB�  �!G�@�Q�AA�Bk�HB��)                                    Bx��f  �          Ae녾�G�A  A3\)BL�HB��׾�G�@��AFffBnG�B���                                    Bx��$  �          Ac�?p��@��
A<��B`z�B�W
?p��@\AM�B��qB�aH                                    Bx��2�  	�          Ac�
>���A�RA733BV�\B���>���@�AH��Bw�B��                                    Bx��AX  
�          Aep��uA&�RAp�B-z�B��׽uAffA3�BNp�B��q                                    Bx��O�  	�          Ad��?
=qA.�\A�HB (�B�\)?
=qA�
A*=qB@��B��                                    Bx��^�  
Z          Afff>���A3�A�
B�RB��>���AG�A(  B;\)B�\)                                    Bx��mJ  T          Ab�H?
=qA
{A0(�BOp�B��)?
=q@�\)AA�Bo�B�.                                    Bx��{�  "          Ah�ÿJ=qAEp�@�A��RB�W
�J=qA2=qAp�B��B�#�                                    Bx����  
�          Af�\���
AF=q@�A��
B�uÿ��
A4  A�\B
=B�k�                                    Bx���<  �          Ad�ÿ�33A4  A
=B�
B�8R��33A�HA"�RB5��B��                                    Bx����  "          AdQ���A7�AQ�B�HB����A#\)Az�B-\)B��                                    Bx����  T          A`z�5A1A  Bz�B�Q�5AG�A33B5�B�B�                                    Bx���.  
�          Af�H�G�AG�@��A�ffB�ff�G�A6�HA\)B\)B�\                                    Bx����  
�          Ak��&ffA[
=@�
=A��BɊ=�&ffAN�\@љ�A�\)B���                                    Bx���z  �          Ah�׿˅A^ff@|(�A{�B�{�˅AS�
@�G�A��
B��q                                    Bx���   �          AmG����Abff@xQ�Ar�\B�(����AW�
@�  A��RB���                                    Bx����  �          Amp��
=Aap�@���A|(�B��
=AV�\@�z�A��B���                                    Bx��l  
�          Ao��*=qAe��@L(�AD��B����*=qA\��@��\A�
=B�Ǯ                                    Bx��  T          Ap���Dz�Ab�R@~{Au�B�z��Dz�AX(�@��\A�ffBͣ�                                    Bx��+�  �          Ap���Z=qAX��@�  A��RB�G��Z=qAK�@�  A�B�                                      Bx��:^  �          Ar{�W
=A[
=@��HA�  Bϙ��W
=AM�@�33A���B�L�                                    Bx��I  �          As
=�!G�A_�@�ffA�{B�ff�!G�AR=q@�A�\)Bɮ                                    Bx��W�  �          Ap�׿�=qAM�@��HA�=qB�8R��=qA<Q�A33B��Bų3                                    Bx��fP  
(          Apz���AM��@��
A�B\���A<  A�Bp�B��f                                    Bx��t�  "          Aqp�����AM@���A��B��q����A;�
A=qB�B���                                    Bx����  T          Ar�\��RA[
=@��HA��Bƅ��RAL��@���A�  B���                                    Bx���B  	.          Ap���>{A\  @��HA�Q�B�W
�>{AO\)@��A߮B���                                    Bx����  �          Ap  �.{A`Q�@��A�B����.{AUG�@ǮA�
=B�\                                    Bx����  �          Am���,��A]G�@���A�=qB�{�,��AR{@��
A�G�B�=q                                    Bx���4  
�          Ajff�5�AP��@���A���B̅�5�AC
=@�A�33B�.                                    Bx����  
Z          Am�4z�A_�
@��A���B��4z�AU�@��\A��HB���                                    Bx��ۀ  �          Alz��E�A^=q@���A}�B��E�ATz�@�\)A�=qB�#�                                    Bx���&  
Z          Ak33�G
=A_�@Q�AN=qB�{�G
=AW�@�\)A��\B�                                    Bx����  �          Ak��s�
A_�@(�A�B�z��s�
AY�@�z�A�
=B�\)                                    Bx��r  T          Am���{AaG�?Tz�@Mp�B�����{A]�@"�\AffB�u�                                    Bx��  T          Al  ��Q�A^�H>\?��HBٽq��Q�A\z�@z�A�B��                                    Bx��$�  "          Al  ���A_
==L��>8Q�B�\���A]G�?�p�@׮B�Q�                                    Bx��3d  
�          Am���=qA`�׿�=q��p�B�
=��=qAa�?�R@��B���                                    Bx��B
  
�          Am���\)A\  �<���7�B�����\)A`(������\)B�Q�                                    Bx��P�  "          Ao\)���AU��Q����\B�=q���A\z��(���#
=B�{                                    Bx��_V  �          Ao���Q�ATz�������B�Q���Q�A[�
�C�
�=G�B�
=                                    Bx��m�  	�          Apz�����AQp����
���B�Q�����AZ�\��G��yp�B�                                    Bx��|�  T          Ao�����AM����=q����B��H����AW
=�������HB�(�                                    Bx���H  �          Ao���{AMG���\)��{B�\��{AV�\���R��z�B�W
                                    Bx����  �          An�R���
AJ�R������B�����
AT  �����=qB��                                    Bx����  
�          Ao����\AM��  ����B�{���\AV�H������B�ff                                    Bx���:  �          Ao���Q�AO�
��ff���B�B���Q�AX�������
BڸR                                    Bx����  "          Anff��33AF{��33���B�\��33AO\)������B�\                                    Bx��Ԇ  
(          Ao33���\AN{��z����B������\AV�\�z=q�rffB�ff                                    Bx���,  �          An�\���AP����  ���HB������AX(��P���J�\B�B�                                    Bx����  #          Al����ffAL  ����(�B�\)��ffATQ��~�R�y��B�                                    Bx�� x  S          Ak���(�AD����  ��  B�#���(�AM���33��B�=q                                    Bx��  	�          Ak�
��33A@����p���\)B�R��33AI������=qB癚                                    Bx���  
�          Aj�\��{ADQ�������B�3��{ALz��\)�|��B��H                                    Bx��,j  �          Aip����RAF=q��p���\)B�aH���RAM���c33�`��B�                                    Bx��;  �          Ah(�����AI���(����\B�L�����AO��@  �?33B���                                    Bx��I�  �          Aj�R��{AP  �n�R�k�B�L���{AU��z��B�G�                                    Bx��X\  �          Aj=q��z�AO\)�tz��r{B�
=��z�AT�����z�B���                                    Bx��g  T          Aj�R��{AR=q�N�R�K
=B�Ǯ��{AV�\������z�B���                                    Bx��u�  �          Ak���z�AU���^�R�Z�RB�#���z�AZ=q��
� ��B�L�                                    Bx���N  "          Ak�����AUp��4z��0��B�
=����AX�ÿ�����B�aH                                    Bx����  �          Al(���\)AV=q�33��B�\)��\)AX�ÿh���a�B��
                                    Bx����  !          Ad  ��AV{>aG�?^�RBڊ=��ATz�?˅@�p�B���                                    Bx���@  T          Aa��z�AUp�?�G�@�(�B�ff��z�AR�\@�A�RB��)                                    Bx����  
�          Aa���=qAQp�?O\)@S�
B�W
��=qAO
=@Q�A�B�Ǯ                                    Bx��͌  
�          A^�R���
ANff>�
=?�  B�Q����
AL��?�(�@�33Bߣ�                                    Bx���2  S          A]����AN�\>�?�B�Ǯ��AMG�?�z�@��
B�                                      Bx����  	�          A]���33AN{>�Q�?�  B�33��33ALz�?У�@أ�B߅                                    Bx���~  "          A^�\��Q�AJff?&ff@,(�B�����Q�AHQ�?��@��B�aH                                    Bx��$  T          A^ff���AMp�?޸R@�Bހ ���AI��@>�RAF=qB�33                                    Bx���  �          A]���  AK\)?��H@�33B��f��  AG�
@<(�AC�B��                                    Bx��%p  !          A_33��=qAK�@p�A"�RB�z���=qAF�H@j�HAtz�B�k�                                    Bx��4  T          A`Q���ffAK�@EALQ�B�z���ffAF{@�G�A���Bߔ{                                    Bx��B�  �          Ab�H���RAM��@<(�A?�B�#����RAHQ�@�z�A��B�33                                    Bx��Qb  �          AO���=qA733@vffA��RBߊ=��=qA0��@��A���B�                                      Bx��`  S          A=�a�A#�
@�{A�p�B��a�A(�@�(�A�B�                                    Bx��n�  �          A((�@:=q�O\)A33B�� C���@:=q��33ABtp�C��3                                    Bx��}T  
�          A*=q@'
=���A!��B�(�C��@'
=�-p�A=qB�Q�C��
                                    Bx����  �          A*{@O\)?��A�
B�z�A�@O\)>ǮA ��B��@��                                    Bx����  T          A(  @>{>�A\)B�\)A\)@>{��
=A\)B�u�C��q                                    Bx���F  T          A$  ?�33?��A\)B�k�A�
=?�33>#�
A Q�B�z�@��\                                    Bx����  �          A+�?�\?�p�A&�HB�\B\)?�\?��A(Q�B�A�
=                                    Bx��ƒ  �          A'�?�=q@�RA"�\B��{B��=?�=q?�=qA%�B�(�BL{                                    Bx���8  T          A$z�>�33@��\A�Bp{B��>�33@�G�A�B�B���                                    Bx����  
�          A$����
A�@��A���B҅��
@�\)@�=qB��B�Q�                                    Bx���  "          A!G�����@�=qA Q�BSG�B�� ����@��A(�Bf��B��                                     Bx��*  �          A"{�h��@�G�A\)BY�B�Ǯ�h��@�=qA
�RBm33B�B�                                    Bx���  
�          A
=��p�@�G�A33Bi�
B�\)��p�@��Ap�B|��B�Ǯ                                    Bx��v  �          A����
@�Q�Az�Bb{BԀ ���
@���A
=BtffB�B�                                    Bx��-  �          AG���33@���@�=qB=qB����33@Ϯ@�(�B�\Bԣ�                                    Bx��;�  
�          A�����H@�z�ٙ��*�\B�Ǯ���H@�  ������p�B�                                      Bx��Jh  T          A�����RA\)��ff�*ffB�G����RAG�������(�B�=                                    Bx��Y  T          A  ��(�@�>�p�@(�B�Q���(�@�(�?�G�@���B��3                                    Bx��g�  �          A{��z�@�(�@G�AI�C����z�@�{@#33A�C��                                    Bx��vZ  T          A%�����@�ff@��HA�  C8R����@���@�  B�HCn                                    Bx���   
�          A$z���(�@ҏ\@�{A�  C����(�@�p�@�{B 
=C��                                    Bx����  
�          A#���=q@ۅ@�ffB ffB�����=q@��@��RBQ�C��                                    Bx���L  T          A ����33@ʏ\@��\B�\C �
��33@��@ə�B�C�                                    Bx����  
�          A
=��G�@θR@�p�Bp�B�  ��G�@��@���B&(�B���                                    Bx����  $          A���k�@�G�@a�A�=qB����k�@�  @���A��B�
=                                    Bx���>             A4(��W
=A�H?���A'�
B�=q�W
=AQ�@#�
Ah��B��H                                    Bx����  	�          A@z��UA2�R�0  �T  B�L��UA5p���z���\B���                                    Bx���  
�          AC��s�
A4���\)�<��B�B��s�
A7
=��33��G�B���                                    Bx���0  
�          ADz�����A1녿=p��\(�B�����A2ff=�\)>�33B�{                                    Bx���  �          AH�����A6�R�AG��_33Bۙ����A9��� ��B�                                    Bx��|  
�          AN�R���\A4  �����B�����\A8���w
=��
=B�{                                    Bx��&"  
�          AMG����HA-�������  Bߊ=���HA3�������
B�Q�                                    Bx��4�  "          AHz��u�A&�\��G��ܣ�B�aH�u�A,��������z�B�{                                    Bx��Cn  T          AMp���
=A-���z���p�B�����
=A2�H��(���=qB�Ǯ                                    Bx��R  T          AF�\��p�A%�������
=B����p�A+
=��{���
Bޞ�                                    Bx��`�  �          A5���(�A �������-�B�33��(�A����
=�z�B���                                    Bx��o`  �          A5녿��@�
=�\)�@�BǞ����A Q���ff�1�B�L�                                    Bx��~  
�          A7�
�Ǯ@�z���H�<�B�{�ǮA
=����.  Bɔ{                                    Bx����  �          A>{��(�A�����833B�{��(�A	�� (��)z�Bʙ�                                    Bx���R  
�          A8����@�z���\�:�B�z���A�R�����,�BД{                                    Bx����  �          A<zῳ33@��
�  �A��B��쿳33A�\�  �3z�B�u�                                    Bx����  �          A�H�B�\@fffAG�Br�C8R�B�\@C�
A��B|�HC��                                    Bx���D  "          A'
=�8��@��A��B^B�\)�8��@���Ap�Bk{B�#�                                    Bx����  T          A6�\���A�H@�\)A܏\B�\)���Ap�@��A�Q�B��                                    Bx���  	�          A8z����
A�
@a�A��B������
A  @��
A�z�B��f                                    Bx���6  "          A9�c�
A.�\?�\A��B�.�c�
A,z�@�AAp�Bؔ{                                    Bx���  
Z          A1���
=A��@,��Ab�\B�k���
=A�R@R�\A�z�B�.                                    Bx���  �          A0z���\)A��?�{Az�B�(���\)A�H@�AM��B�R                                    Bx��(  T          A/�
���\A�R�Z=q��\)B�{���\A�7��v�RB�.                                    Bx��-�  
�          A-���Tz�A  ���مB܀ �Tz�AQ��������RB�ff                                    Bx��<t  T          A.�\��A�׿���:�HBꞸ��A��    ���
B�=                                    Bx��K  "          A8(���A�
�xQ���
=B�����A33�W�����B��
                                    Bx��Y�  "          A@z��+�@�z�����>G�Bڣ��+�A���\�2=qB؅                                    Bx��hf  h          A?
=�G�@�(�����Mz�B�\�G�@�(����A  B�L�                                    Bx��w  
�          A'��Q�@�\)��z��#�\Bٳ3�Q�@����  ��
B�{                                    Bx����  
�          A(��)��@�G����\�B�Ǯ�)��@�����p���z�Bٔ{                                    Bx���X  
�          A"�H��Q�@��H�����7�BϽq��Q�@�R��p��,33B�W
                                    Bx����  
�          A���=q@����
�n�B�ff��=q@�\)��\)�b�
B��f                                    Bx����  "          A33?�
=?!G��	�\)A��?�
=?���z��qBG�                                    Bx���J  �          A�H?z�?:�H�
{§��BO33?z�?��	G�¢�RB~�R                                    Bx����  
�          A=q����@�  ���
�
=B�������@�Q�������B�L�                                    Bx��ݖ  
�          A	��Q�@\��ff�JffC޸��Q�@����  �(z�C�                                     Bx���<  �          A����Q�@�  @�Q�A�
=CT{��Q�@���@���A�(�C	s3                                    Bx����  "          A���'�>\)@�33B�  C0��'��k�@�33B��fC9�                                    Bx��	�  
�          A	G��?��@��B�8RC��?�G�A ��B�G�C��                                    Bx��.  
�          A��0��@�Q�@���BY\)B�(��0��@l(�@޸RBa�B���                                    Bx��&�  
�          @�=q�0��@�G�@���BD\)B����0��@q�@�=qBL�
B�B�                                    Bx��5z             @���G�@���@�33B<�B��)�G�@���@��BFQ�B�(�                                    Bx��D   
�          A���\)@��R@�  B:z�B�=�\)@�{@�\)BC�B�Ǯ                                    Bx��R�  T          @�(��h��?�ff@�  B7��C W
�h��?���@���B;{C"�\                                    Bx��al  
�          @�G����?���@�p�B�z�C�=���?�\)@�
=B��C5�                                    Bx��p  "          @�p���33?:�H@�{B�8RC���33?�@�
=B��3C$�q                                    Bx��~�  "          @���*=q?��@�G�B{�C(=q�*=q>�{@���B|�C,�=                                    Bx���^  �          Aff��(�@<(�@�BF
=C���(�@,(�@�G�BK
=Cs3                                    Bx���  �          @�33���
@=q@��BA��Cs3���
@
�H@�ffBE��C�=                                    Bx����  R          A�R����?�z�@�=qBA��C+�����?�@�z�BD��C!�{                                    Bx���P  �          @�����z�?��H@�\)BH(�C%c���z�?}p�@���BJ(�C'��                                    Bx����  T          @�=q�\��?
=q@�
=B}��C+#��\��>�\)@߮B~��C/k�                                    Bx��֜  �          @���XQ�>�(�@ڏ\B}�HC,�q�XQ�>8Q�@��HB~��C0�R                                    Bx���B  �          A z���ff@
�H@�{BV��C����ff?�Q�@У�BZp�C@                                     Bx����  T          A�����?��@�ffBY�C#�\���?���@׮B[��C&��                                    Bx���  �          Aff��Q�@��R@�p�B33C@ ��Q�@���@�=qB  CY�                                    Bx��4  
�          A ����33@��\@%�A�C5���33@��@/\)A��C�q                                    Bx���  T          A����\)@�p�>�
=@>{CO\��\)@���?(�@��Cff                                    Bx��.�  �          AG���@�Q�?+�@�z�B�����@�\)?h��@���B�                                      Bx��=&  
�          Ap���ff@�Q�?   @@��Cz���ff@�?:�H@�z�C�\                                    Bx��K�  �          Ap��ָR@�(��h�����
Cp��ָR@��Ϳ5��z�CT{                                    Bx��Zr  T          A�\��Q�@����Q����C(���Q�@�=q�#�
�qG�C\                                    Bx��i  �          AG��ۅ@�(��"�\��  C���ۅ@��R�Q��s
=C��                                    Bx��w�  �          A\)��(�@����
� ��CB���(�@����\)��\)Cz�                                    Bx���d  T          A���p�@�Q�������Cz���p�@�z����H�Q�C�q                                    Bx���
  T          A  ���@�
=���
�	=qC�=���@�33������C
=                                    Bx����  
�          A���=q@�\)���\�   C&f��=q@�(����Q�CQ�                                    Bx���V  	�          A���\)@���33�G�B�(��\)@����ff�\)B��                                    Bx����  "          @��
�}p�@�=q�xQ���z�B����}p�@�p��p  ��G�B��f                                    Bx��Ϣ  
�          @�(���H@�p������7ffB�L���H@�������2G�B�
=                                    Bx���H  T          @������?�����Q��pffC.���?�(���
=�m=qC=q                                    Bx����  "          @����
=��
=?�{A<(�Cx�q��
=��  ?}p�A((�Cx�{                                    Bx����  "          @���������\?:�HA{C�Uþ������H?(�@ڏ\C�W
                                    Bx��
:  T          @�=q�(������@��A�C�H��(����=q@A�C�Q�                                    Bx���  "          @������{?�Aq�C~ff�����R?��A_33C~}q                                    Bx��'�  
�          @��ÿ��
���\?��A�=qCx�R���
���
?��A�p�Cx޸                                    Bx��6,  �          @��
�u?������B�LͿu?��������p�B�=                                    Bx��D�  
�          A�R=L��@陚��(��z�B�G�=L��@�������{B�G�                                    Bx��Sx  �          A���(�@����\)�"G�B�k��(�@أ�����{B�B�                                    Bx��b  
�          A  ���@�����(��+=qBə����@�z���  �'33B�B�                                    Bx��p�  �          A�Ϳ�Q�@أ����R�33B�𤿸Q�@ۅ��33�\)Bˣ�                                    Bx��j  "          @���{@���  ��z�B���{@�z��
�H��33Bя\                                    Bx���  �          @G���녽���?ٙ�B`\)C9\��녾�?ٙ�B`  C:ff                                    Bx����  
�          @h���Q�0��@)��B;z�CD
�Q�:�H@(Q�B:z�CD�R                                    Bx���\  �          @�Q���ÿ�
=@p  BUz�CR�=��ÿ�p�@n�RBSCS��                                    Bx���  "          @�G������@��RB��CF����R@��RB�.CH@                                     Bx��Ȩ  �          @�
=��Q�?&ff@�=qB��)C��Q�?�@ʏ\B�p�C�=                                    Bx���N  �          @�p�?0��>�33@ÅB��=A׮?0��>�\)@ÅB��A�=q                                    Bx����  
�          @�Q쿏\)<��
@ٙ�B�G�C3���\)�u@ٙ�B�B�C7(�                                    Bx����  V          @޸R�0��?^�R@��HB��=C J=�0��?L��@�33B�� C�{                                    Bx��@  R          @Ϯ<��
?��
@ƸRB�B�B��<��
?�(�@�
=B�p�B���                                    Bx���  �          @�z�?@  ?�(�@�ffB���B|=q?@  ?�@�
=B���Bx�                                    Bx�� �  $          @�  ��33?
=@�=qB�#�B𞸾�33?
=q@�=qB�B��=                                    Bx��/2  
�          @�(����H���H@�33B��
CDͿ��H���@�33B��\CE�
                                    Bx��=�  $          @��
�\���@�=qB���CIn�\�W
=@�=qB���CP�                                    Bx��L~  T          @�zῬ�Ϳ�Q�@��B��fCg�=���Ϳ�  @�Q�B�G�ChaH                                    Bx��[$  
�          @��H>�p���z�@��B��C�y�>�p����H@陚B�k�C�Z�                                    Bx��i�  
Z          Aff�(�����@���B�Q�C}��(���33@�Q�B��RC~                                    Bx��xp  V          @��\�!녿p��@ᙚB�{CH^��!녿xQ�@�G�B��HCI�                                    Bx���             @�
=��(��u@ϮB]z�C@W
��(��}p�@ϮB]=qC@�                                    Bx����  T          A(��j=q��@�{Bt�CNQ��j=q��@�{Bt�\CN�f                                    Bx���b  T          A
�\�g
=�+�@��
Bj\)CX�\�g
=�,��@�Bi��CXǮ                                    Bx���  �          A=q�����Vff@陚BU{CY�R�����W�@�G�BTCZ�                                    Bx����  �          A���|���vff@�G�BW(�C`B��|���w
=@���BV�C`W
                                    Bx���T  "          A�
�AG����\@�\)B>��Cq�
�AG����H@�\)B>��Cq��                                    Bx����  �          A�R�333��ff@޸RBS��Cm�\�333��ff@޸RBS��Cm��                                    Bx����  �          A��{���Q�@�B9�Cgٚ�{���  @�B9��Cg�{                                    Bx���F  "          Aff�J=q��G�@��B�Cu  �J=q����@�=qB(�Ct��                                    Bx��
�  �          A  �+���R@���B33Cy�H�+���ff@��B��Cy�)                                    Bx���  �          AG��$z���z�@\)A�Q�Cz�{�$z���(�@�Q�A�p�Cz�\                                    Bx��(8  �          A��5����@p  A�p�Cw���5��z�@qG�AָRCw�)                                    Bx��6�  �          A(���p���Q�@:�HA�{C�����p���  @<��A�C��                                     Bx��E�  
�          A녿�R��\)@߮BJ
=C��=��R��ff@��BK
=C��f                                    Bx��T*             A����p���
=@��
B9C|O\��p���{@���B:�
C|5�                                    Bx��b�  �          AG��G��Ӆ@�p�B ��Ct���G��ҏ\@ƸRB!��Ct�)                                    Bx��qv  �          A� ����@�
=A�
=CW
� ������@���A��
CJ=                                    Bx���  T          Ap������G�@���B{C��������  @��HB�C��                                     Bx����  T          Az��33�Ӆ@���B 
=C~  ��33��=q@�=qB!�RC}�f                                    Bx���h  "          A���33���@�33BP{Cs���33���@�z�BQ��CsB�                                    Bx���  
�          A
=�;����\?�G�A=�Cy���;���=q?�=qAE��CyxR                                    Bx����  �          A=q�.{��G����
�  Cs���.{���H����  Ct�                                    Bx���Z  �          @��H�B�\������  �D(�C����B�\��33���R�A�
C���                                    Bx���   
�          @��H>��R����^{�p�C��>��R��{�Z�H���C��=                                    Bx���  �          @��H?\)�o\)��(��>{C�Ff?\)�r�\���H�;z�C�8R                                    Bx���L  �          @�Q�>�\)�����
¨L�C�3>�\)�
=q���§{C���                                    Bx���  �          @�ff>�=q?�33��z���B��>�=q?������ffB�z�                                    Bx���  "          @�{@9�����H@S�
A��C�y�@9����G�@W�A�Q�C��
                                    Bx��!>  T          @�Q�?Q녿�ff@U�B��{C��?Q녿�  @VffB��C���                                    Bx��/�  T          @��+�?�33@���Bq\)Cff�+�?�p�@�  Bo��C(�                                    Bx��>�  �          @���J=q?�ff@��BiffC!���J=q?���@�z�Bh=qC J=                                    Bx��M0  �          @���.{��@�z�B�ffC4���.{=u@�z�B�ffC2��                                    Bx��[�  �          @�(��\)�xQ�>��H@�C��\�\)�w�?\)A
=C��\                                    Bx��j|  �          @����\)@%�@j=qBC�B�녿�\)@)��@g
=B@z�B���                                    Bx��y"  "          A=q�`  @Ϯ@}p�A�B��`  @��@uAڏ\B�.                                    Bx����  
�          A
ff����@��@w�A��B������@�\)@o\)A��B��                                    Bx���n  �          A��hQ�@�=q@(�A}p�B�  �hQ�@ۅ@�
Am�B�R                                    Bx���  T          @���.{@I��@�33BEp�C� �.{@P  @�G�BB  C��                                    Bx����  �          @�33���H@��@tz�A��C �����H@�{@mp�A��C @                                     Bx���`  �          @�R���H@2�\@�\)B/�CQ����H@8��@�p�B-G�C\)                                    Bx���  
�          @�G����H@��H@dz�A�{C����H@��@]p�AԸRC�f                                    Bx��߬  �          A
=�x��@�=q?�=qA{B���x��@�33?�A (�B�                                    Bx���R  
(          @�ff��(�@�\)@  A�G�C����(�@���@Q�A�G�CaH                                    Bx����  �          @�p���z�@�{?��
A?33B���z�@�\)?��A-��B�u�                                    Bx���  T          @�\��ff@�=q@J=qAҸRC:���ff@���@C33A���C�                                    Bx��D  �          @�{��Q�@��@�z�B�C���Q�@#�
@��\B�HC�                                     Bx��(�  T          @�
=�z�H@O\)@p��B��Ch��z�H@U@j�HB�C�)                                    Bx��7�  
�          @�p����R@%�@^�RB�HCJ=���R@*�H@Z�HB  Cu�                                    Bx��F6  T          @�33��33@��@%�A�G�C p���33@z�@!G�A�33C�f                                    Bx��T�  �          @�{��
=@G�@   A���C(���
=@@(�A��\C��                                    Bx��c�  �          @�Q���ff@��\@J=qAǙ�C0���ff@��@C33A��C�)                                    Bx��r(  
&          @�Q�����@�
@g
=A�ffC"�����@
=q@c33A�z�C!��                                    Bx����  �          @�=q��33?�
=@j�HA�RC#����33@�\@g
=A��HC"ٚ                                    Bx���t  
�          @�Q����>��
@J�HA�\)C1E���>��@J=qAʏ\C0xR                                    Bx���  	�          @����33��z�@C�
A�{C=����33����@EA�ffC=.                                    Bx����  
�          @��
����\)@.{A�z�CH  ����
=q@2�\A�33CGO\                                    Bx���f  T          @��\����?
=@z�A��C/ff����?(��@33A��C.�
                                    Bx���  
�          @�z���{@
=@�A�p�C#����{@�@
=Az�\C##�                                    Bx��ز  
(          @�=q��  @ff?�Q�Af�\C#޸��  @
=q?�\)A^=qC#k�                                    Bx���X  
�          @�R��z�?���@-p�A��C'����z�?��@*=qA�=qC&��                                    Bx����  �          @�z���
=@Mp�@{A���C����
=@Q�@
=A�C)                                    Bx���  �          @�����p�@A�@
�HA�
CT{��p�@Fff@�
As33C�
                                    Bx��J  �          @�\)���@��@��A��
C �����@{@�A�z�C �                                    Bx��!�  
(          A����H@0  @33Aj�HCn���H@4z�?���A_\)C�3                                    Bx��0�  	�          A����@;�?���AQG�C����@?\)?�(�AD��C��                                    Bx��?<  
�          A ����\@U?�\AK�
CǮ��\@Y��?�33A=��C^�                                    Bx��M�  �          A���@N{?���A%p�C���@QG�?�=qA�C��                                    Bx��\�  �          A�R���
@)��?��\@�C �\���
@,(�?k�@���C ��                                    Bx��k.  �          A\)��\)@(�?��AffC"s3��\)@\)?��HA�
C"!H                                    Bx��y�  
�          A�\� (�@\)?���A�
C$ff� (�@�?�\)@�z�C$�                                    Bx���z  �          A{���?�p�?��
@��C%�\���@G�?s33@ָRC%�=                                    Bx���   
Z          @�z����
@&ff@z�At  C�����
@+�?��HAg33CW
                                    Bx����  T          @�{���H@ ��@#33A�{C xR���H@'
=@��A�C�                                     Bx���l  T          @�
=��Q�@8��@�{B#ffC33��Q�@Fff@�=qBCxR                                    Bx���  T          A�
��p�@   @#�
A��HC!aH��p�@&ff@��A��RC ��                                    Bx��Ѹ  �          AG���?��R@H��A��
C$�q��@�@C�
A��RC$�                                    Bx���^  �          A(���
=@#33@Mp�A���C!+���
=@+�@FffA�ffC @                                     Bx���  T          A�
���R@5@>{A���CǮ���R@=p�@5A�  C��                                    Bx����  
�          A	����p�@:�H@#�
A���C.��p�@AG�@�A���CxR                                    Bx��P  T          A�R��p�@,��@33Ab�HC �
��p�@2�\?�
=AU�C �                                    Bx���  
�          @�(���Q�@ ��?޸RAL  C ����Q�@%?�\)A>ffC aH                                    Bx��)�  
�          @����z�@33?}p�@陚C"����z�@?c�
@У�C"n                                    Bx��8B  
�          @��H���?��?�p�A.�\C'�����?ٙ�?�z�A%p�C':�                                    Bx��F�  
�          A����p�@Fff@
=Ax��C޸��p�@L(�?��HAg33C8R                                    Bx��U�  T          A ����=q@��@p  A�\)C(���=q@��\@`  A���CJ=                                    Bx��d4  	�          Az����\@�(�@J�HA�G�C	ff���\@���@:�HA�ffC�)                                    Bx��r�  �          A�
���H@�33@33Af�RC
���H@�{?���AO�
C��                                    Bx����  	`          A{��p�@}p�?�=qA3\)C0���p�@�G�?��A�C�q                                    Bx���&  
Z          AQ���\@vff?�ffAI�C� ��\@{�?�{A4Q�C��                                    Bx����  
�          A���p�@s33?��A(  C�H��p�@w�?���A
=Cp�                                    Bx���r  
�          A���G�@i��?��\A
{C.��G�@mp�?��@��
C�\                                    Bx���  �          A
=��\)@c�
?�{A0(�C���\)@h��?�
=A  C
=                                    Bx��ʾ  
Z          A�\����@s�
?u@�  C�3����@vff?B�\@��
C�                                    Bx���d  "          A�
���H@r�\?��A8(�C�)���H@w�?�Q�A!��C\)                                    Bx���
  
�          A=q�ڏ\@�Q�?�z�AX��C�
�ڏ\@�33?ٙ�A@Q�C�                                    Bx����  �          A33���H@�Q�?��A�C)���H@�=q?�A�C��                                    Bx��V  
Z          A{��
=@h��@6ffA�G�Cs3��
=@q�@)��A��
C��                                    Bx���  
�          A���G�@}p�@33Aip�C� ��G�@��?�=qAPQ�C�                                    Bx��"�  
�          A
=���R@�\)�����{C#����R@��H�.{���
C�                                    Bx��1H  
�          A �����@���
=��\)C
=���@���fff��ffC=q                                    Bx��?�  
�          A Q���ff@�=q�
=q�x��C
0���ff@�G��W
=��  C
c�                                    Bx��N�  T          AG���
=@��Ϳ:�H��{C\)��
=@�33�����C��                                    Bx��]:  "          A z���@���>#�
?��C�\��@��ͽ��\(�C��                                    Bx��k�  
(          @��R�θR@��\?B�\@�\)C��θR@��
?   @eCxR                                    Bx��z�  "          A z���p�@�ff?W
=@��C�
��p�@��?�@��
C��                                    Bx���,  
�          Ap����@�
=��z�� Q�B�\)���@�z����+\)B�                                      Bx����  	�          A����\@�\)�\)�y��C�����\@��k���C                                    Bx���x  
�          A  ���R@����  ��G�Cz����R@�z�z���=qC��                                    Bx���  �          AQ���\)@��>�(�@=p�C}q��\)@�=q>\)?}p�Cc�                                    Bx����  
�          A�\��p�@��\>��?޸RC
��p�@��\��\)��C\                                    Bx���j  R          A�\�׮@��\?�z�A,(�CaH�׮@�p�?���A��C�                                    Bx���  "          A����H@��
?�(�A\)C�����H@�ff?���@�C+�                                    Bx���  
�          A�����
@��?�z�A\)C����
@�=q?��@ָRC��                                    Bx���\  
Z          A  ��@�ff?�=qA;33C����@��?�  ACG�                                    Bx��  �          A����@�{@
=Atz�C����@��\@ffAXz�C:�                                    Bx���  "          Aff��=q@�z�?��RAJffC� ��=q@�  ?ٙ�A,(�C޸                                    Bx��*N  
X          Ap���Q�@ٙ�?��@���C����Q�@��
?aG�@�(�C��                                    Bx��8�  �          A(����@׮?�(�A
�HC�����@�=q?��@�z�C^�                                    Bx��G�  �          A=q��p�@�(�?�{A�C(���p�@�
=?�@���C��                                    Bx��V@  T          Az���  @ָR?�{@��RC+���  @�G�?k�@���C�)                                    Bx��d�  "          A����Q�@�
=?�33@أ�C���Q�@���?333@��HCٚ                                    Bx��s�  �          A��׮@�z�?�=q@�z�C\)�׮@�ff?!G�@mp�C!H                                    Bx���2  T          A  ���@��
?�  @�(�Cs3���@�{?aG�@�C)                                    Bx����  T          A  ��Q�@�녿Q�����C(���Q�@����(����Cz�                                    Bx���~  �          A33��p�@���>��?s33C����p�@��þ�\)���
C��                                    Bx���$  	`          A����Q�@��H>�ff@0��C
=��Q�@�33=#�
>uC
�R                                    Bx����  �          A���Q�@��
>�z�?�G�C
���Q�@��
���J=qC
��                                    Bx���p  
�          AG���@��?�@Mp�C�{��@�=�?=p�Cz�                                    Bx���  �          A�����@У�?���@�=qC!H����@ҏ\?.{@�33C�)                                    Bx���  T          AG�����@ҏ\?�G�@�G�CaH����@�(�?��@S�
C#�                                    Bx���b  T          A��\)@θR?�G�@�=qC	:���\)@���?O\)@��C��                                    Bx��  "          A�R���@ҏ\?��\@�
=CO\���@�z�?\)@Z=qC{                                    Bx���  �          A  ��{@��
?�@��C#���{@�{?.{@��
C޸                                    Bx��#T  
�          A�����
@ָR?�G�@�Q�C�����
@���?J=q@�=qC8R                                    Bx��1�  
�          A���أ�@���?+�@��HC�)�أ�@�>W
=?��CxR                                    Bx��@�  "          A�R�Ϯ@ٙ�>L��?���C���Ϯ@�G���z��\C��                                    Bx��OF  �          A�����@�z�?ٙ�A(Q�C
=����@�  ?��
@��C��                                    Bx��]�  �          A�H��z�@��@   AJ=qC\)��z�@�{?�{A"ffC��                                    Bx��l�  
�          Ap���{@�33?h��@��C\)��{@���>�G�@3�
C#�                                    Bx��{8  "          A{��p�@�G����
��B��f��p�@�Q�&ff���
B�\                                    Bx����  "          A{��(�@��
�\�
=B���(�@�=q�s33��\)B�Q�                                    Bx����  T          A(����A���G����HB� ���A��\)�:{B�#�                                    Bx���*  
Z          AQ��N�RA��>�R���\B�W
�N�R@�(��dz���(�B��\                                    Bx����  
�          AQ����@�33�dz����B�{���@����(���B�\)                                    Bx���v  �          A
=�w�@�=q��33�D��B���w�@���\)��G�B��                                    Bx���  
�          A�~{@������=B�3�~{@�������{33B�                                    Bx����  V          A���{�@��ÿ�Q��=qB�(��{�@���� ���W�
B���                                    Bx���h  �          A���p�@��þ�G��:=qB��q��p�@�
=�z�H��B�(�                                    Bx���  �          A����z�@�{�#�
���B�����z�@��
����(�C 
                                    Bx���  V          A(���  @ҏ\�B�\��ffC+���  @�  ��G��Q�C��                                    Bx��Z  �          A���z�@��ÿz��r�\C
=��z�@�
=�����\CQ�                                    Bx��+   
�          A�
��(�@�
=�s33����C.��(�@�(���(��ffC�{                                    Bx��9�  T          A
�H��
=@����z����RC����
=@ə���z��0��C5�                                    Bx��HL  �          A33���@�p���(�� ��C�\���@ə���(��6ffCT{                                    Bx��V�  �          A�
���R@�\)�\���CQ����R@ҏ\�33�S�C�                                    Bx��e�  S          A���=q@�ff�����33B����=q@�33�����$  B��R                                    Bx��t>  T          A�R��@��k����HC ^���@�z�L����G�C ��                                    Bx����  T          A�����@��ý��G�B�33���@�  �5���RB�u�                                    Bx����  "          A(����@��ÿ���vffC�\���@ָR��33��\C�                                    Bx���0  �          A����(�@��ÿ#�
����C �3��(�@޸R������z�C �q                                    Bx����  
�          AG���
=@�\)�8Q����C\)��
=@��Ϳ��
�ffC��                                    Bx���|  �          A33��z�@�Q쿇�����B�����z�@��Ϳ��'�
B���                                    Bx���"  �          Aff���@ᙚ�6ff��33B�\)���@�G��Z=q���B�aH                                    Bx����  T          A���33@��G
=��  C �3��33@����i����33C�f                                    Bx���n  �          A=q���
@���Z�H���
CG����
@�Q��|�����HC��                                    Bx���  �          A���H@����Dz����RC!H���H@��
�e���Cc�                                    Bx���  �          Ap���G�@�(�>#�
?��\B��3��G�@���(��0  B�Ǯ                                    Bx��`  �          A{��@�z�?G�@�p�C�q��@�>L��?��C��                                    Bx��$  "          A=q���@��>��@%�C�f���@ڏ\�#�
��G�C�)                                    Bx��2�  W          A=q��p�@޸R>��?�33Ch���p�@޸R���
�G�Ck�                                    Bx��AR  �          Ap����@�z�>�?Q�B����@��
�   �HQ�B��)                                    Bx��O�  	`          Aff����@�  >B�\?�  B�������@���
=�-p�B��H                                    Bx��^�  
+          A����(�@�z�L�Ϳ��
B����(�@�33�Y����z�B�k�                                    Bx��mD  
�          A���\)A����ff����B�R��\)A���G��,Q�B�ff                                    Bx��{�  
�          A33���@�
=?O\)@�{B�����A (�>�?8Q�B�ff                                    Bx����  W          A�����@�(�>�
=@{B�L�����@�zᾅ���ffB�B�                                    Bx���6  
�          A�
��  @�(�����g�B�L���  @������
��  B��
                                    Bx����  
�          A
=��Q�@�{�}p���ffB�#���Q�@��\��
=�(Q�B��)                                    Bx����  
�          A���@�
=�333��{B�녿��@�ff�]p����B�                                    Bx���(  %          A�?\)@�z���������B��
?\)@���{�ffB�ff                                    Bx����  �          A�
�#�
Aff�mp���
=B�
=�#�
@���������B�p�                                    Bx���t  
�          A{�+�A�����lz�B����+�A��G
=���RB�                                    Bx���  
�          AG����\A�\��
=��  B�����\A Q����;�B�q                                    Bx����  
(          A����{@�{?�33A(�C����{@�G�?G�@��HC�\                                    Bx��f  
�          A��{A   �(��e�B��f��{@��������\)B�z�                                    Bx��  
�          AQ����
A�\��\�'
=B� ���
A�� ���n�\B                                    Bx��+�  
�          Ap���ff@�������HCn��ff@�p������(�C�                                    Bx��:X  "          AQ����Ap�������\)B�q���A\)���0Q�B�                                     Bx��H�  �          A�
���@�z�xQ����RB������@��׿�
=�$��B�p�                                    Bx��W�  �          A(���Q�@�녿\(���{Cn��Q�@�ff��G���RC�
                                    Bx��fJ  T          A{���@�\���R��B�Q����@����8��B�aH                                    Bx��t�  T          A(�����A녿����(�B�{����@����\�'�
B��f                                    Bx����  
�          A�\����@��^�R��ffB�u�����@�=q�˅�(�B�=q                                    Bx���<  �          Az���\)@��;��5B����\)@�=q������\B�u�                                    Bx����  �          A�H��ff@���    �#�
B�ff��ff@�  �5���B���                                    Bx����  4          A�����\@���u��33B�����\@��\�G����\B�aH                                    Bx���.  �          Ap�����@�  =��
>��HB�
=����@�\)�!G��s�
B�8R                                    Bx����  �          A����p�@��;���#33B�����p�@��\�����G�B�B�                                    Bx���z  �          A����  @�녿Q���33B�G���  @�ff����!G�B��                                    Bx���   �          A���p�A녿p�����B�  ��p�@����Q�� ��B���                                    Bx����  �          Az���\)@�33�W
=����B�����\)@�\)��ff��B�k�                                    Bx��l  �          A���\)A�\�����G�B�Q���\)A�������G�B��                                    Bx��  �          A!���A��?�\)@У�B�Ǯ���A	>��
?�B�ff                                    Bx��$�  �          AG���  @�=�Q�?z�B��)��  @��Ϳ&ff��G�B�
=                                    Bx��3^  �          A�����AQ���0��B�p�����A��fff��B�                                    Bx��B  �          A(���ff@�G���33���B�����ff@�\)�����Q�B�B�                                    Bx��P�  �          A\)��(�@陚�W
=��=qB�B���(�@��fff��Q�B���                                    Bx��_P  �          AG���G�AG����
��33B��)��G�A Q쿊=q��Q�B�G�                                    Bx��m�  �          A�����R@�ff?
=q@K�B�L����R@�
=�W
=��  B�33                                    Bx��|�  �          Aff��
=@���.{���\B����
=@��fff���RB�z�                                    Bx���B  �          A���\)@�p�=�\)>ǮB����\)@�z�(���z�HC �                                    Bx����  �          A
=�Ǯ@��>�?J=qC
�Ǯ@�z�z��Z=qC+�                                    Bx����  �          A
=��@ᙚ���H��RC\)��@��Ϳ���4(�C��                                    Bx���4  �          A�R��
=@��>#�
?xQ�C\)��
=@ᙚ���J=qCn                                    Bx����  �          A�H����@�\>�@>{B�������@��H�k���Q�B��f                                    Bx��Ԁ  �          A���@��
=���?#�
C }q���@�33�
=�l(�C �
                                    Bx���&  �          A������@�p���ff�333C �
����@�33�����C#�                                    Bx����  �          AQ���(�@񙚿޸R�%�B�����(�@�33�{�k
=C ��                                    Bx�� r  �          A���  @�Q��(��6{C�\��  @����+��z{C��                                    Bx��  �          A�\���
@�����C�����
@�\)�
=�\��Cu�                                    Bx���  �          A����
@������,Q�C�{���
@����   �n�\C��                                    Bx��,d  �          A=q��=q@����H�ip�C�3��=q@�G��Fff��ffC�\                                    Bx��;
  �          A����
@�G��Q��c�B�ff���
@���G
=���B�ff                                    Bx��I�  �          A\)��G�@����7
=��p�B��{��G�@��H�dz���33B���                                    Bx��XV  T          A�
��33@�  �<(���\)C ����33@��g���=qC�q                                    Bx��f�  
�          A\)��\)@����a���G�C 33��\)@ȣ���p���Q�C�
                                    Bx��u�  
�          AQ���{@��
�J=q��  B��=��{@����u����B�8R                                    Bx���H  T          A�W
=@��
��=q�  B��)�W
=@Å��ff�  B���                                    Bx����  �          A���g�@У��������B��g�@��������G�B�                                    Bx����  �          A{���H@�=q���k(�B�����H@e��=q�
BÙ�                                    Bx���:  �          AG����@�{��
=�XG�B�����@�
=��p��np�B�\                                    Bx����  �          A%G���33A��1G��x��B�����33@����dz���B��R                                    Bx��͆  �          A#33��z�A (��1��}��B�L���z�@�ff�c�
���
B�z�                                    Bx���,  �          A!����(�@�\)�@������B�����(�@�z��r�\���\B�L�                                    Bx����  �          A$����
=A��7
=��G�B�=q��
=@����j�H��33B�ff                                    Bx���x  �          A%G����H@�\)�S�
��=qB�\���H@����H��p�B���                                    Bx��  �          A���\@e���ff��CxR�\@Fff������RC                                      Bx���  �          A	���=q@!G��������Ck���=q@���  ��C!�                                    Bx��%j  T          A\)����@�(��j�H���C\)����@�\)��Q���CE                                    Bx��4  �          A���(�@�G������͙�B�W
��(�@�33����\B�                                      Bx��B�  T          A������@ٙ��k����HB�G�����@�����33��G�C ٚ                                    Bx��Q\  �          A\)��=q@�G��`  ���
CaH��=q@�����p����C�                                    Bx��`  �          A�����R@ҏ\�Q���33C�����R@ƸR�{��ǮC�{                                    Bx��n�  T          A����33@���c�
��ffC  ��33@�������ϙ�C�                                    Bx��}N  �          A  ��z�@�ff�G����\C�)��z�@�33�p�����C33                                    Bx����  �          A�
���\@�z��<(���=qCE���\@љ��g���G�C��                                    Bx����  �          Ap���(�@�z��@����z�C :���(�@ٙ��mp�����C��                                    Bx���@  �          A������@�
=�S33��{C�\����@�33�~{�ɅC&f                                    Bx����  �          A�����@�(��\(���p�C+�����@Ǯ���H��
=Cٚ                                    Bx��ƌ  �          A����=q@�p��5���C �
��=q@��H�a����
C0�                                    Bx���2  �          A�H����@�(��\)�[33C ������@��
�<�����C
=                                    Bx����  �          A����
@�Q��&ff�}C����
@ָR�S33��ffC(�                                    Bx���~  �          A   ��\)@�  �s�
��G�C W
��\)@�=q�����ׅC�                                    Bx��$  �          A!���@����z���(�C!H���@�=q���H���C!H                                    Bx���  �          A#
=��
=@�
=���R�܏\CY���
=@θR��z�� {C�                                    Bx��p  T          A!�����@�z���33��\C ����@˅�����Q�C�                                    Bx��-  �          A�����@�=q�(��Z=qB��H���@���;�����B��H                                    Bx��;�  �          A"{��{@�p��!��hQ�B�\)��{@��
�Tz���  B�u�                                    Bx��Jb  �          A'�����Ap��1��w
=B�L�����@�Q��fff���HB��{                                    Bx��Y  �          A(���ə�A Q��+��l  C ��ə�@��R�_\)����C@                                     Bx��g�  �          A'����A ���P�����HB�{���@�{��=q���\B��3                                    Bx��vT  �          A(  �ۅ@��R���R�.�RC���ۅ@�
=�1G��tz�C�{                                    Bx����  �          A(����Q�@���N{C\)��Q�@����H�����CaH                                    Bx����  �          A&�H�\A��H�V�RB����\@�=q�O\)���RB���                                    Bx���F  �          A&�H���AQ��b�\����B����@�������=qB�aH                                    Bx����  �          A&=q��=qA���R�\����B�L���=q@�p����
����B�Ǯ                                    Bx����  �          A%G���=qA\)�C33���B��)��=qA���z=q���
B�\                                    Bx���8  �          A&�\��  A���L(�����B���  A�H������33B��)                                    Bx����  �          A+���  Aff�ٙ��ffB�q��  A�H�(Q��c�
B���                                    Bx���  �          A.�H��
=A���z���p�B�����
=Az����I�B��                                    Bx���*  �          A8�����A
=�W�����B뙚���A�����H��(�B홚                                    Bx���            A<z�� ��A���b�\����B�
=� ��A
{��ff����B�p�                                    Bx��v  �          A?��UA$��������B���UA  ��33��ffB���                                    Bx��&  �          A=p��'
=A�H��33���RB�(��'
=A  ��=q�{B�Q�                                    Bx��4�  �          A<������A!����p����
B�u�����A���{�ӅB�k�                                    Bx��Ch  �          A=����
A!�<#�
=L��B�k����
A Q쿂�\��G�B�                                    Bx��R  �          AB=q��  A/33?��@�{B�\��  A0  ��Q��(�B��)                                    Bx��`�  �          AB{���RA/�?�@�
=B�u����RA1��?
=q@#33B�                                    Bx��oZ  �          A@���W
=A6�R���
��{BԽq�W
=A4(�����)��B�33                                    Bx��~   �          A<z��p  A/33����B��
�p  A+\)�:�H�g33Bڣ�                                    Bx����  �          AB{����A4(���33��z�B݀ ����A2�R��G���B���                                    Bx���L  "          AD(�����A-�?:�H@[�B�{����A.=q�Ǯ��=qB�                                      Bx����  �          AB�H����A4(�?
=q@%�B޸R����A4(���R�:�HB޸R                                    Bx����  �          ABff�'
=A:ff���	B�8R�'
=A6�\�@  �eB�                                    Bx���>  �          AD(��J=qA<  ��G�����B�\�J=qA8���p��9p�BҊ=                                    Bx����  �          AD  �'
=A>=q�z�H���B����'
=A;�����%�B�#�                                    Bx���  �          AD(���A>�\?�R@:=qBɳ3��A>�\�(��7�Bɮ                                    Bx���0  �          AF�H�R�\A7\)���  B�\�R�\A3��@  �f�\BԽq                                    Bx���  �          AG�
���A=���-�B�\)���A8���a����B���                                    Bx��|  �          AIp�� ��AE����{���\Bƅ� ��AB�R���.=qB���                                    Bx��"  �          AK\)�\��A?���H�/�
B�.�\��A:�\�hQ����B�                                      Bx��-�  �          AI�aG�A=�z��*{B���aG�A8���aG���{B��f                                    Bx��<n  �          AJ=q�u�A@  ���
��p�B�aH�u�A=G��  �$Q�B��)                                    Bx��K  W          AH���K�AB{�\)�&ffB�ff�K�A@�ÿ����ʏ\Bє{                                    Bx��Y�  �          AI��$z�AD��>��?�Q�B˨��$z�AD  ��  ���B˽q                                    Bx��h`  �          AI녿�AG\)?L��@j=qBĊ=��AG�����	��Bą                                    Bx��w  �          AIp��"�\AD  ?�@�
=B�p��"�\AEG�>#�
?8Q�B�L�                                    Bx����  �          AI�	��AE��?�ff@�B�Ǯ�	��AF�R<�>��BǮ                                    Bx���R  �          AJff��\)AH  ?�
=@�(�B�� ��\)AH�ý��;�ffB�p�                                    Bx����  �          AI�X��A;�@<��AYp�B�B��X��A?�?޸R@�ffBӣ�                                    Bx����  �          AH���:=qA=�@>�RA\(�BϞ��:=qA@��?�G�A�B�{                                    Bx���D  �          AHz��`��A9��@AG�A_�Bծ�`��A=��?���A{B�                                      Bx����  �          AH  �Dz�A;
=@%�A@��B�u��Dz�A>ff?���@�z�B���                                    Bx��ݐ  �          AG��#�
AC
=���
��p�B˽q�#�
AA녿����\B��H                                    Bx���6  �          AH(��J=qA@Q�?���@�=qB�z��J=qAAp�=�Q�>�(�B�L�                                    Bx����  �          AG�
�/\)A?�@��A"=qB�Ǯ�/\)AB{?xQ�@�ffB�p�                                    Bx��	�  �          AH  �E�A>�H@�\A{B���E�AA�?O\)@o\)BЙ�                                    Bx��(  �          AH(���
A@��@�A.�RBɸR��
AC�?���@�ffB�ff                                    Bx��&�  �          AH���,(�A@��@�A Q�B�(��,(�AC\)?s33@�33B��
                                    Bx��5t  �          AFff��AB�R?��@���B�ff��AC��L�Ϳk�B�W
                                    Bx��D  �          AF{��AB=q?�  @��
B���AB�H�u���Bǳ3                                    Bx��R�  �          AB{���A=�?u@�=qB��
���A=�k���{B�Ǯ                                    Bx��af  �          A>ff�)��A5G�@G�A0Q�B�W
�)��A8  ?�{@�(�B���                                    Bx��p  �          A;\)�{A%G�����
=B�L��{A����ff��=qB͏\                                    Bx��~�  �          A9p��c�
A'�
�S33����Bي=�c�
A!�����H��z�B��)                                    Bx���X  �          A=�fffA%����{���\B�W
�fffA�����R��{B��                                    Bx����  �          A<(��.{A$Q����R��{Bѳ3�.{A����R��z�B�L�                                    Bx����  �          A8���(Q�A%���  ��(�BЙ��(Q�A�������ڣ�B��                                    Bx���J  �          A<���mp�A/
=��(����Bـ �mp�A-p��\��=qB���                                    Bx����  �          AC�
��\)A0��@+�AJ�RB����\)A4(�?Ǯ@��HB�\)                                    Bx��֖  �          AC�����A.ff@#33A@��B�p�����A1��?���@�=qB��                                    Bx���<  T          AB�\���RA+33@�
A1�B�R���RA-�?�p�@��
B�                                    Bx����  �          A>�R����A"�R@i��A��\B� ����A'�@'
=AJ�HB�#�                                    Bx���  �          A<������A
=@�z�A��\B�.����A ��@J=qAx��B�k�                                    Bx��.  �          A9���x��A)G�>k�?�Q�B�\)�x��A(�ÿQ����B�u�                                    Bx���  �          A7
=��G�A(Q�?s33@��B�ff��G�A(�ý�Q��B�=q                                    Bx��.z  �          A9���\A%�@(�AB=qB�����\A(��?�33@�{B��f                                    Bx��=   T          A8���mp�A*�H?��A�
B�B��mp�A-�?Tz�@�(�B���                                    Bx��K�  �          A7\)�S�
A(  @>{AqG�B����S�
A+�
?�A�HB�=q                                    Bx��Zl  �          A6{����A"=q@�Aď\BÊ=����A(��@j=qA�B��                                    Bx��i  �          A2�R���Aff?�ffAG�B�aH���A(�?
=@E�B��H                                    Bx��w�  �          A+���{A��@J=qA��
B�����{A�@��AC\)B�u�                                    Bx���^  �          A-��  A��?n{@��B㙚��  A=#�
>aG�B�aH                                    Bx���  �          A1�z=qA
=�9���xz�B����z=qA�u��B�aH                                    Bx����  �          A/33�G
=A#�
@�A8��B��
�G
=A&ff?�@�B�W
                                    Bx���P  �          A5p��0��A)�@0��Ac33B�=q�0��A,��?�(�A��BЮ                                    Bx����  W          A5G��^{A&�H@(��AW�B��^{A*=q?�{A�HB��                                    Bx��Ϝ  	+          A5��N{A,  ?�\A33B�L��N{A-�?8Q�@g
=B���                                    Bx���B  �          A5����A#�
��  ���B�W
���A!�����R�#
=B��                                    Bx����  �          A5p���p�A�R��\)��
B�����p�A\)�#33�P(�B��f                                    Bx����  �          A2�R���
A�\�����\)B������
A
=�/\)�eG�B��                                    Bx��
4  �          A1��ffA z�G�����B�=q��ffAff�޸R���B�Ǯ                                    Bx���  "          A3�
��Q�A Q���:�HB�(���Q�A�R���
���HB鞸                                    Bx��'�  T          A2=q����A�R=��
>�
=B�G�����A{�Q���  B�z�                                    Bx��6&  �          A2=q���RAz��G���B��)���RA����\����B�#�                                    Bx��D�  �          A1����\A��(���[�B�R���\A\)�˅�z�B�=q                                    Bx��Sr  �          A0z����A  �#�
�Y��B�Ǯ���A
=������
B�{                                    Bx��b  
Z          A/�
�`��A"�H?���A\)B�\�`��A$��?!G�@Tz�Bٳ3                                    Bx��p�  "          A)p���G�Ap���\)�p�B����G�A=q�!G��]p�B�                                    Bx��d  �          A)����\A
�R���FffB��H���\A	��������HB�p�                                    Bx���
  
�          A)��33A�H>��
?�p�B����33A�R�
=�O\)B���                                    Bx����  �          A*=q����A���
��G�B�Ǯ����A�ÿn{����B�
=                                    Bx���V  �          A&{�l��Az�?   @2�\B�ff�l��A�׾����{B�aH                                    Bx����  �          A&{�%A
=����'�B�(��%A����33���B�ff                                    Bx��Ȣ  "          A-G��>�RA$�Ϳ��
���B�Q��>�RA"ff��p��)�BԽq                                    Bx���H  "          A0���0  A&{�p��:ffBѳ3�0  A!��J�H��Q�B�ff                                    Bx����  �          A5�\)A'\)�w
=���HB�#��\)A ����=q��ffB�{                                    Bx����  �          A6�H�Q�A   ���\��\)B��Q�A�
��\)����B�G�                                    Bx��:  	�          A6�R��  A&�R�\����z�B����  A ����z���(�B�                                    Bx���  �          A5��p�A/33��33�B�L��p�A,  �)���X  Bͽq                                    Bx�� �  �          A6�H��
A0Q����\)Bˣ���
A,���8Q��ip�B��                                    Bx��/,  �          A6�R��  A.�\�5�f�RB�33��  A)�u����RBƸR                                    Bx��=�  �          A0Q�Tz�A"�H�������
B�G��Tz�Az�����p�B���                                    Bx��Lx  �          A+�
�L��A���G����B�ff�L��Az�������B��                                    Bx��[  "          A+��0��AG���ff����B���0��A�\��=q��(�B�W
                                    Bx��i�  
�          A+��k�A�\�\)����B��{�k�A(�������
B�                                    Bx��xj  �          A-��W
=A"�\�a����RB�uÿW
=A�������
B���                                    Bx���  T          A+33�#�
A"�\�Mp����
B�=q�#�
AG���33���HB�z�                                    Bx����  �          A)����A ���QG���  B�B���A��������HB�p�                                    Bx���\  �          A
=�(��A=q?�\@\(�B��f�(��A�\�W
=��z�B��
                                    Bx���  �          A"�\�y��@��H@˅B��B�W
�y��@�(�@�
=B	�
B�                                    Bx����  �          A=q�L(�@�33@�\)A�B�B��L(�A(�@���A��
B�B�                                    Bx���N  �          A.ff�XQ�A�
@���A���B�L��XQ�A33@�Q�A�33B�aH                                    Bx����  
�          A"=q��G�A(�?aG�@��RB�u���G�A��=��
>��B�=q                                    Bx���  "          A ���`  Ap���z��33B�
=�`  Azῌ�����HB�L�                                    Bx���@  
�          A���HA�ÿ�����HB�ff��HA�R��(��6�HB���                                    Bx��
�  �          A33��A��Y����p�B�W
��A\)���  BϨ�                                    Bx���            A�\�{A�H�%��r�HB���{A�R�W
=���B���                                    Bx��(2  �          A�R����A=q�P����B�
=����AG�������Q�B��)                                    Bx��6�  
�          A!����A{�.�R�~ffB�ff��A�a�����B�{                                    Bx��E~  T          A$Q��(�Aff���\��p�B����(�A  ���?33B�Q�                                    Bx��T$  "          A$�׿�=qA�H�&ff�m�Bƅ��=qA�H�Z=q���B�                                    Bx��b�  
�          A"�R�1�A�
��  ����B�G��1�A����\)���B���                                    Bx��qp  �          A"�\�,(�A
=�h����Q�B�p��,(�A�������G�B׽q                                    Bx���  �          A#���A��<��
>\)B�{��AQ�8Q�����B�(�                                    Bx����  	�          A&=q��{A�
@�Q�A�z�B���{AG�@n�RA�z�B�(�                                    Bx���b  T          A,(�<#�
Ap�@�p�A�p�B��H<#�
Az�@�z�A�33B��H                                    Bx���  �          A.ff�J=qA��@�ffA�p�B��׿J=qAQ�@��A˅B�8R                                    Bx����  
�          A'��0��Ap�@�{AԸRB�ff�0��A
=@y��A���B��                                    Bx���T  "          A1G���A"ff>8Q�?k�B�����A"{�(���\(�B��H                                    Bx����  
�          A1����A �ý�Q��ffB�W
���A Q�h�����B�                                    Bx���  Q          A)p���(�Az���33�݅B̨���(�A	����=q�  B��
                                    Bx���F  "          A'��5A��P�����
B�aH�5A�H������p�B�ff                                    Bx���  �          A(  �E�Aff�����=qB�.�E�A�����\��G�Bڙ�                                    Bx���  
�          A'33�%AQ���p���=qB���%A������Bր                                     Bx��!8  
Z          A'\)�J�HAz���ff���
B۳3�J�HA{��z���=qB�u�                                    Bx��/�  �          A&=q�EAG��W
=��G�B؞��EAz����H���
Bٽq                                    Bx��>�  T          A&=q�2�\AG��_\)���B�=q�2�\AQ���
=����B�L�                                    Bx��M*  T          A(Q��|��Aff��p���\)B�Q��|��@����=q���B�{                                    Bx��[�  �          A$���e�@��������B�q�e�@�
=����
=B�B�                                    Bx��jv  T          A"�R��
=A	�/\)�|  B����
=A��Y�����B�G�                                    Bx��y  �          A!����A  �$z��m��B����AQ��N{���
B�                                      Bx����  T          A!G����A
=q���R�6ffB� ���A\)�)���tQ�B�                                    Bx���h  
�          A!����A�R�X������B��H����@�z���Q����RB�                                    Bx���  
�          A�
��{Ap��i����z�B�p���{@�G���  ��
=B�G�                                    Bx����  
�          A ������AG�����.{B�������A
�\�#33�lz�B���                                    Bx���Z  %          A\)��z�A\)���
��B垸��z�A�\�}p�����B��)                                    Bx���             A��_\)AQ�?�
=A!�B�k��_\)A�?�G�@�=qB���                                    Bx��ߦ  T          A���XQ�A�׿�=q�4z�B�33�XQ�A{�p��t  B���                                    Bx���L  T          AG��b�\A��?�ff@�p�B��b�\A��>\@p�Bޅ                                    Bx����  
�          A�
�HQ�A?��R@�p�B���HQ�A�H?��@I��Bخ                                    Bx���  
�          A!G��h��A��?�ff@��Bޮ�h��A{?(�@_\)B�aH                                    Bx��>  
Z          A!��1�A  @?\)A��B�p��1�A
=@�
AS\)B�Ǯ                                    Bx��(�  �          A"�H�2�\A
=q@��A�G�B����2�\A33@z�HA���B֨�                                    Bx��7�  
�          A"ff�S33A
�H@{�A�p�Bݔ{�S33A
=@R�\A�z�B�u�                                    Bx��F0  �          A=q�4z�A	p�@~�RA�  B�W
�4z�A��@VffA���B�W
                                    Bx��T�  T          A"=q�1G�AG�@1�A~�HB�\�1G�A  @
=A@Q�B�z�                                    Bx��c|  �          A!녿xQ�A
=@(Q�Aqp�B�k��xQ�A?�Q�A0��B�8R                                    Bx��r"  "          A�
�\)A
=q@:=qA��
B�\�\)A�@�\AU�B��                                    Bx����  �          A ���s33A��@:�HA��B�q�s33A�@33ATQ�B��f                                    Bx���n  �          A"{�^{A��@;�A�ffB���^{A�@�\AQp�B�.                                    Bx���  
�          A!p��p  A��@N{A�
=B�.�p  A�
@'
=Ap  B�B�                                    Bx����  
�          A&�\�0  A33@S�
A��HB�p��0  Aff@*=qAm��B�Ǯ                                    Bx���`  f          A*=q��(�A�\@���A�Q�B���(�A�R@Z=qA�{B�z�                                    Bx���  �          A*ff�b�\A�@�Q�A���B����b�\A�@XQ�A��B���                                    Bx��ج  �          A&�H��A
�R@|��A�Q�B���A�\@W
=A��RB�G�                                    Bx���R  �          A&�R��=qAz�@!G�A_�B�W
��=qA�R?�33A'�
B�3                                    Bx����  T          A,����Q�A�\?��@���B����Q�A�?:�H@y��B��
                                    Bx���  �          A)��hQ�A�?�{A  Bܙ��hQ�A�\?xQ�@��B�L�                                    Bx��D  �          A*�H�R�\A!G�?��@��B�33�R�\A"=q?(��@aG�B���                                    Bx��!�  �          A#\)���RA ��>aG�?�G�B��f���RA �׾�
=���B��f                                    Bx��0�  �          A$�Ϳ�G�A"{?E�@���B��ÿ�G�A"�\=�?(��B��                                    Bx��?6  �          A%����A ��?.{@s�
B�.���A!�=#�
>uB��                                    Bx��M�  �          A"{��=qA�?ٙ�A33B�8R��=qA�\?�=q@�z�B�\                                    Bx��\�  �          A"�\����A�?���@�=qB�����A z�>�ff@"�\B��)                                    Bx��k(  �          A%p����A ��?�{@�33B�.���A!�?=p�@��B�
=                                    Bx��y�  �          A$z��G�Aff?�(�A  B��H�G�A�?\(�@�=qBͳ3                                    Bx���t  �          A#
=�{A33�:�H���BД{�{A=q������G�B�                                    Bx���  �          A"�H�{A  ��{��(�Bӊ=�{A����R��RBԊ=                                    Bx����  �          A"ff�Q�Aff��Q���{B�ff�Q�A���  �
G�Bр                                     Bx���f  �          A#
=�AG�@�{?J=q@�
=B���AG�@�
=>\@6ffB��f                                    Bx���  �          A(���W�@�{@��B+�\B��
�W�@��H@׮BffB�L�                                    Bx��Ѳ  �          A&�H�\)A33@�p�B
��B��\)Az�@�A���Bԅ                                    Bx���X  �          A&=q�g
=@�G�@�=qB\)B���g
=@���@�z�Bp�B�                                    Bx����  �          A&=q�w�@�{@�{A���B�
=�w�A�@�\)A�\)B�aH                                    Bx����  �          A%����@��@�  B (�B�z����A z�@���A�\)B�3                                    Bx��J  �          A#�
�]p�A�@�p�A�
=B���]p�A�
@�ffA�\)B�\)                                    Bx���  �          A!����33@��
@�\)A�z�B�{��33A�@�G�A��
B鞸                                    Bx��)�  �          A ���<(�A��@O\)A���B��<(�A33@/\)A}B�k�                                    Bx��8<  �          A{�/\)Az�@333A�Q�B����/\)A�R@�
AX��B�Q�                                    Bx��F�  �          A33�A�Ap�@�Q�A���B�
=�A�AG�@�=qAϮB��                                    Bx��U�  �          A Q��J�HA{@���A��Bޔ{�J�HA�@�33A��B�u�                                    Bx��d.  �          A��E@��@��RB�HBߔ{�EA Q�@�G�A�G�B�8R                                    Bx��r�  �          A#
=�,(�A�
@`��A�{B�Q��,(�A�\@B�\A�33BԽq                                    Bx���z  T          A$���5�A�?�{A(�B�p��5�A��?���@�p�B�8R                                    Bx���   �          A��^�RA
=@(�AM��Bފ=�^�RA��?�p�A"{B��                                    Bx����  �          A�����A�?���@�33B�\����Az�?k�@�33B�q                                    Bx���l  �          A!��{@��?u@�ffC \��{@�ff?\)@L��B��
                                    Bx���  �          A$����  A�?8Q�@�  B�B���  A(�>���?�B��                                    Bx��ʸ  �          A%����p�A��?z�@N{B�B���p�A�>��?W
=B�(�                                    Bx���^  �          AG����
A���\)�`  BǙ����
A��*=q��B��)                                    Bx���  �          A�����A
=q�L������B�Ǯ����A��fff��z�B�{                                    Bx����  �          A녿n{A���7
=����B�uÿn{A�\�P����Q�B���                                    Bx��P  �          AG��+�AQ��ff�V�\B��+�A
�R�   ��z�B��H                                    Bx���  �          A��z�A�׿�33�*�\B�33�z�A33��\�S33B�u�                                    Bx��"�  �          A
=q���RA�
��ff�&{B�����RA�R���N=qB�aH                                    Bx��1B  �          A�����A
=����o
=B�W
���AG��'���33BѸR                                    Bx��?�  T          A��s�
@��R�\���B��H�s�
@��H�g���
=B��
                                    Bx��N�  �          AG��Vff@�=q�.{���B����Vff@�ff�C33��B�                                     Bx��]4  �          A����\@���5��(�B�W
��\@����J�H����B��
                                    Bx��k�  �          A�
��(�@����  �
=B�ff��(�@�G���G��G�B�aH                                    Bx��z�  �          A��p�@����\)�3
=Bʙ���p�@�z���\)�<\)B˔{                                    Bx���&  �          Aff�Y��@����=q�5Bî�Y��@�������?
=B�\)                                    Bx����  �          A��Ǯ@��
��{�B�\B����Ǯ@�33��p��KB�                                      Bx���r  �          A�\�$z�Aff�0  ���RB����$z�A ���Dz���  B�p�                                    Bx���  �          A=q�\)A�׿��
��\B�aH�\)A������6�RB�q                                    Bx��þ  �          A�H�aG�A33>�Q�@{B�{�aG�A\)<�>8Q�B�\                                    Bx���d  �          A��4z�@��@��
B�B�aH�4z�@�33@��B{B�B�                                    Bx���
  T          A���*�H@�  @�=qB#33B�3�*�H@�
=@�=qBQ�B�\)                                    Bx���  �          AG��333@�
=@�=qB��B�G��333@�@�=qB�HB�\                                    Bx���V  T          AQ��)��@�p�@��A�p�B�B��)��@��H@��HA�  B�u�                                    Bx���  �          AG��n{@���@�G�A�z�B��n{@�{@�G�A�(�B�                                    Bx���  �          A�����H@�Q�@��\A���B�q���H@���@u�A���B��
                                    Bx��*H  �          A��u@�33@eA���B���u@�
=@U�A���B�u�                                    Bx��8�  �          A����A�\?�z�Az�B�8R����A33?���@��
B���                                    Bx��G�  �          A33����A�ÿ�R�dz�B�\����Az�c�
��z�B�8R                                    Bx��V:  �          A!p���
=@��>�33@�C� ��
=@�=q=�G�?(��CxR                                    Bx��d�  �          A%p���
=@�p�?��HA��C.��
=@�
=?��RAz�C��                                    Bx��s�  �          A
=��
=@��@(�ALz�CT{��
=@�
=?�p�A8��C\                                    Bx���,  �          A�����@θR@Q�AH��C	�����@У�?�
=A6=qC	G�                                    Bx����  �          A ����@�ff?���@�G�C޸��@�\)?aG�@�G�C                                    Bx���x  "          A"=q��ff@�=q=��
>��C� ��ff@�=q���B�\C��                                    Bx���  T          A"ff��
=@���������
C
@ ��
=@��
�����
=C
k�                                    Bx����  
�          A z����H@����C33��C�R���H@�{�L����\)C!H                                    Bx���j  �          A���=q@�G��0����{C� ��=q@�
=�:�H��  Cٚ                                    Bx���  �          A�\����@�ff�P������C������@ۅ�\(����C��                                    Bx���  �          Az����@���[�����B������@�
=�g����
B�z�                                    Bx���\  �          Aff��\)@��H�?\)����CJ=��\)@���J=q���C�
                                    Bx��  �          A����\)@���/\)��z�C�{��\)@�ff�:=q����C�)                                    Bx���  �          A���|(�@�p�����ᙚB��H�|(�@�zΰ�
��
B�
=                                    Bx��#N  �          A�
��z�A�
@P��A��\B�aH��z�A��@C33A�ffB�33                                    Bx��1�  �          A!�����A�\@n{A��B��Ϳ���A�
@`��A��Bʞ�                                    Bx��@�  �          A����  @����l(���{B�z���  @�ff�vff��=qB�                                    Bx��O@  �          A��ƸR@�33�]p���Q�CW
�ƸR@����e��
=C��                                    Bx��]�  �          A�H���H@�����\)��\)C�����H@����33��C+�                                    Bx��l�  �          A�\�^�R@����{�L�B��H�^�R@�����G��O�RB�k�                                    Bx��{2  �          A��XQ�@�z���\�D33B�8R�XQ�@�  ���GB��=                                    Bx����  �          Az��O\)@�(�����!{B�Ǯ�O\)@ȣ���p��$�RB�{                                    Bx���~  �          A���R@�33��33�>Q�B�����R@�\)��ff�A��Bޔ{                                    Bx���$  �          A(��$z�@�����
=�:��B����$z�@�{����>  B��                                    Bx����  �          Ap��0  @�z�����{B�=q�0  @����Q��Q�B���                                    Bx���p  �          A��fffA Q�>��
@33B��fffA z�>L��?���B�                                    Bx���  �          Az��A\)�
=q�b�\B͞���A33�(�����\Bͣ�                                    Bx���  �          A��(�Aff>��?z�HB�G��(�Aff=#�
>k�B�G�                                    Bx���b  �          A\)��ff@�p���{�/
=Cz���ff@��H��  �1=qC�                                    Bx���  �          A�R���R@�����ff� C �����R@�ff��Q��"�HC �q                                    Bx���  �          Aff�n{@����{���HB�\�n{@ۅ������p�B��f                                    Bx��T  �          AG��q�@������ 
=B�{�q�@��
��
=��B�z�                                    Bx��*�  �          A��J�H@��p����G�B�=�J�H@�=q�u��p�B�                                    Bx��9�  T          AG��s�
@ڏ\�����B�G��s�
@�G���(���p�B�=                                    Bx��HF  �          A	p��
=q@�\)�L�����B�녿
=q@��R�QG���\)B���                                    Bx��V�  �          A\)?��@����B�\��z�B�.?��@��
�G
=���B��                                    Bx��e�  �          A	�?z�@�z��w
=��{B�G�?z�@��z=q��33B�B�                                    Bx��t8  �          AQ�\(�@�z�������\B�LͿ\(�@����H��\)B�W
                                    Bx����  �          A�
���
@���~�R��  Bϔ{���
@�  ������z�BϨ�                                    Bx����  �          A	���{@�p����R���HB�{��{@�����  ��
=B�(�                                    Bx���*  �          A�R�&ff@���mp���{B�\)�&ff@ᙚ�o\)��B�p�                                    Bx����  �          A
=�Y��@�����=q����B��\�Y��@�(����H���B���                                    Bx���v  �          Az�s33@�\��{��\B��s33@�=q���R���B�Ǯ                                    Bx���  �          A
ff>�\)@�
=����
B��{>�\)@�R��{�=qB��{                                    Bx����  �          A�?�p�@�
=��\)���B��?�p�@�
=����33B��                                    Bx���h  �          A(�?��
@�\)�����\)B��?��
@�
=�����p�B��                                    Bx���  �          A��>���@�  ���H�ޏ\B��\>���@�  ���H��ffB��{                                    Bx���  �          A�
>#�
@�z�������\B��>#�
@��������G�B��                                    Bx��Z  �          A
=q>�G�@�G��������B�(�>�G�@陚��G�����B�(�                                    Bx��$   �          A	�=��
@�33��z���\B���=��
@ۅ���
��B���                                    Bx��2�  �          Az�:�H@ڏ\������RB�(��:�H@��H��  ��B��                                    Bx��AL  �          AG��u@��H��ff��\BĊ=�u@Ӆ�����BĀ                                     Bx��O�  �          A\)����@�p����R�5z�B�����@�ff��{�4ffB��H                                    Bx��^�  �          A	G��Z�H@�
=�Ӆ�J=qC��Z�H@�Q��ҏ\�I33C Ǯ                                    Bx��m>  �          A(��6ff@�Q���p��Zz�B��3�6ff@�������Y=qB��                                    Bx��{�  T          A��xQ�@����G��7�HC�3�xQ�@�����Q��6��Ck�                                    Bx����  �          A�aG�@�p�����Q�B�Ǯ�aG�@�ff��=q�B�k�                                    Bx���0  �          A��\@��
��p��  B�.��\@�������=qB���                                    Bx����  �          A�
�HQ�@��
��33�1
=B��)�HQ�@�p������/(�B�Q�                                    Bx���|  �          A�
����@�G���p��0�
C������@�33���
�/{CT{                                    Bx���"  �          A	�Q�@G
=����{�B��Q�@L(����
�y�B�G�                                    Bx����  �          A���\)?�Q��ff��C����\)?���{��C�                                    Bx���n  �          A
=q���H?����ff��C	�H���H?��{��C�                                    Bx���  �          A���p�>����ffC"{��p�?
=����HC(�                                    Bx����  �          A  �޸R>k��z��RC,�=�޸R>�33�z�u�C(�                                    Bx��`  �          A
=q���?�R�=q�C �\���?@  �{�fCQ�                                    Bx��  �          A
=q���
?�\)����3C	� ���
?�  �Q�u�C��                                    Bx��+�  �          A�Ϳ^�R@7���
=�qB��Ϳ^�R@@  ��p��B�ff                                    Bx��:R  �          A�׿�p�@��
����]�\B�uÿ�p�@����{�YB֙�                                    Bx��H�  �          Aff��@����=q�&�B����@��H���R�"�RBՙ�                                    Bx��W�  T          A\)���
@�{�����ffB�녿��
@�G������=qBҀ                                     Bx��fD  �          A=q��33@��������.�B؀ ��33@�(������)�
B��)                                    Bx��t�  �          AQ�c�
@�=q��
=�:�\B�#׿c�
@�{��33�5�B�                                    Bx����  �          A����
@�������333B�����
@�������.��B���                                    Bx���6  �          A=q���
@��R��Q��J��B��H���
@�33�����E�B�#�                                    Bx����  �          A
=>���@\(���\� B�>���@g
=��  �}��B��                                    Bx����  �          A33�@  @(Q����R�Bӳ3�@  @3�
��z��BѮ                                    Bx���(  �          A{�:�H@�
=����e(�B�uÿ:�H@�����ff�_��B�Ǯ                                    Bx����  �          A(����@��׿Ǯ�8(�CB����@�녿��&�HC�                                    Bx���t  �          A���=q@�\)@\)AhQ�Cٚ��=q@�p�@�Au�C+�                                    Bx���  �          A  ��{@�ff?�p�A1G�C�)��{@���?�\)A@z�C)                                    Bx����  �          AQ���\)@���?�Q�@�=qC
T{��\)@�\)?���A�C
�                                     Bx��f  T          A
{����@љ����H�QG�C(�����@�녾�����C�                                    Bx��  �          A����@�33�=p���G�C:����@˅�\)�q�C#�                                    Bx��$�  �          A(���Q�@˅�����{C����Q�@�z῁G��ڏ\C�q                                    Bx��3X  �          A  ���H@��
�S33���C W
���H@�
=�G
=��ffB���                                    Bx��A�  �          A\)�\)@�
=��Q��	33B����\)@��
��=q�=qB�p�                                    Bx��P�  �          AQ���G�@θR�p�����B�
=��G�@ҏ\�c33��G�B��                                    Bx��_J  �          A���z�H@��H�����33B�aH�z�H@�\)�|����=qB�L�                                    Bx��m�  �          A
{��  @θR�q���Q�B�����  @��H�c�
��G�B���                                    Bx��|�  �          A
=q���R@�G��Z�H��{B�����R@���L����G�B��                                    Bx���<  �          A	���i��@�
=���\���B����i��@��
���
��  B�                                    Bx����  �          A��C�
@�����33�(�B�33�C�
@�{�����33B�#�                                    Bx����  T          A  �@  @�������z�B� �@  @ڏ\�����p�B�ff                                    Bx���.  �          A��ff@��\��\)�7=qB�3�ff@����Q��/(�B�
=                                    Bx����  �          A	��qG�@�{��33�$�\B�{�qG�@��������Q�B��f                                    Bx���z  �          A	�{@�
=��p��D��B�.�{@��R��\)�<�B��)                                    Bx���   �          A���^�R@�=q��  ��{B����^�R@�\)�p����{B�u�                                    Bx����  �          A
ff��Q�@�
=�h����=qB�k���Q�@�Q�!G�����B�.                                    Bx�� l  �          A33����@���?�\)Ap�B��{����@ָR?�33A4��B��                                    Bx��  �          A
�R�r�\@�@
�HAk�B��r�\@�Q�@\)A�33B�(�                                    Bx���  �          A	��  @��H�+�����B�����  @��
��Q��(�B�k�                                    Bx��,^  �          A	������@��
�c�
���RB�������@���z��y��B�ff                                    Bx��;  �          A���
@��H��  ��B������
@�z�n{��Q�B�G�                                    Bx��I�  �          A�R���@�\)��p���{B�� ���@��ÿh�����HB��                                    Bx��XP  T          A����@�׿��H�(�B�����@�\�������HB�Q�                                    Bx��f�  �          A����@ָR�aG���\)C@ ��@�  �\)�h��C)                                    Bx��u�  T          AG����@�{�W
=��
=B�\)���@�\)�   �P��B��                                    Bx���B  �          A{���@�(�����(�B������@�{��  �У�B�W
                                    Bx����  �          A
=��ff@���
�H�`z�CW
��ff@�������>�HC��                                    Bx����  �          A���  @θR�Ǯ� ��C���  @��ÿ�p�����C�)                                    Bx���4  �          A����p�@�{��Q���C   ��p�@�{>�=q?޸RC                                     Bx����  �          AQ����@�Q�?�ffA#�B��)���@�p�?�AJ�RB��{                                    Bx��̀  �          A��@�(�?��@��C E��@��?�G�A��C �\                                    Bx���&  �          AQ���
=@���?�ffA�\C���
=@�{?�z�ADz�C�                                    Bx����  �          A  ��
=@ָR?��A((�Cn��
=@Ӆ@   AN�\C�)                                    Bx���r  �          A�\���@߮?���A(�B�  ���@���?��HA0��B��                                    Bx��  T          A{��z�@�  ?���@��
C޸��z�@�?�p�AC+�                                    Bx���  �          A�H��\)@��Ϳ�33���C��\)@�
=��G��љ�Cu�                                    Bx��%d  �          Ap���ff@�p����8Q�C =q��ff@��>\@��C G�                                    Bx��4
  �          A{���\@ᙚ?�@[�B�33���\@�Q�?s33@�z�B��=                                    Bx��B�  �          A=q��
=@�=q?�  A�B����
=@�\)?�
=A.=qB���                                    Bx��QV  �          A{����@��?���@�33B������@�{?\A{B��q                                    Bx��_�  �          A���=q@�R?���@�
=B���=q@�(�?��
A�
B�\                                    Bx��n�  �          A����  @��?�33@��B�\)��  @�?���A((�B�                                      Bx��}H  �          Ap�����@޸R?.{@�(�B�������@���?�\)@�  C �                                    Bx����  �          Aff��33@��>��@B�\B�k���33@��
?s33@��
B��q                                    Bx����  �          A�R����@�>L��?��B�Q�����@�R?0��@�{B��                                    Bx���:  �          A
=��33@�=q�����B���33@�=q>aG�?�
=B��q                                    Bx����  �          A�H���@�\)��{�z�B������@�녿c�
���B�aH                                    Bx��Ɔ  �          A������@�R�z��mp�B�  ����@�\)���
�   B��
                                    Bx���,  �          A{���@�>��R@ ��B��3���@�ff?Q�@�=qB���                                    Bx����  �          AG���
=A z�#�
��\)B�\��
=A (�?
=q@P  B�.                                    Bx���x  �          A�\��(�A Q���L��B��3��(�A   ?�@HQ�B���                                    Bx��  �          A(���(�@�z�L�Ϳ��
B�z���(�@�z�>��
@33B��                                     Bx���  �          A�\��@�\������{B�=q��@�z�
=�fffB���                                    Bx��j  �          A�
��\)@��H���O�B�#���\)@�
=�������B�8R                                    Bx��-  �          A�����\@�����H��G�B�(����\@�\)�&ff�x��B���                                    Bx��;�  �          AG����\@�������B������\@�Q�G���B�k�                                    Bx��J\  �          A�
��{@�p������p�B����{@���\)�Z=qB�=q                                    Bx��Y  �          AG����@�{�@  ��  B�33���@�\)�#�
�z�HB���                                    Bx��g�  �          A����@�
=�������B�\���@�\)>B�\?�33B�                                      Bx��vN  �          Aff����@񙚽��Ϳ��B�������@�G�>��H@Dz�B��q                                    Bx����  �          A
=��ff@�녽�\)��B���ff@�?�@N�RB�(�                                    Bx����  �          A�\��=q@�ff�#�
��G�B�\��=q@�{>�(�@+�B��                                    Bx���@  T          A  ��ff@��ü��L��B�  ��ff@�Q�?
=@c33B�#�                                    Bx����  �          AQ����H@�{�B�\����B�\���H@�>�(�@'�B��                                    Bx����  �          A����@���u��(�B�{��@���>\@�\B��                                    Bx���2  �          Az����\@�
=���R���B�����\@�
=>��
?�Q�B��3                                    Bx����  �          A(���Q�@�G���  �\B��q��Q�@�G�>\@G�B�Ǯ                                    Bx���~  �          A������@�
=��\)��G�B�8R����@�ff?\)@[�B�aH                                    Bx���$  �          Az���33@��H�n{��Q�B��3��33@�zᾞ�R���B�L�                                    Bx���  �          A���=q@�
=��(��-p�B�����=q@�\)>B�\?��RB��R                                    Bx��p  �          A\)����@�z��(��'
=B�Ǯ����@�z�>k�?�z�B��R                                    Bx��&  �          A33��{@��R�   �@��B�ff��{@�
=>8Q�?��B�L�                                    Bx��4�  �          AG���  A����  ���B�W
��  A�\��\)��Q�B�                                      Bx��Cb  �          A{��(�Ap��s33��p�B�����(�A=q�k���{B�z�                                    Bx��R  �          Aff��33A (��
=q�L��B�u���33A Q�>8Q�?��B�\)                                    Bx��`�  �          A�����A���(���|��B������A�=�\)>�(�B�Ǯ                                    Bx��oT  �          A{���A Q�\(���z�B�\)���A �þ��@  B�{                                    Bx��}�  �          A\)��33A��5���RB���33A<�>L��B���                                    Bx����  �          A����@�
=���L(�B������@��>W
=?��RB��3                                    Bx���F  �          A�����A녿W
=���B�k����A�R�L�;��RB�33                                    Bx����  �          A����Q�Aff�0�����
B�z���Q�A�H=�G�?(��B�Q�                                    Bx����  �          Ap�����A�H���R��\)B�3����A�R>��H@<(�B�                                    Bx���8  �          A33��(�A(��#�
�s33B�#���(�A�?(��@y��B�G�                                    Bx����  �          A�H��Q�@��Ϳfff��=qB���Q�@�ff���=p�B�k�                                    Bx���  T          A����
@�z�E����B�����
@�p�    <��
B��                                    Bx���*  
�          A  ��z�@��\)�QG�B��
��z�@�{>k�?���B��q                                    Bx���  
�          A(���z�@��(���x��B��
��z�@��R>�?B�\B���                                    Bx��v  �          A(���(�A{�\)�Q�B�  ��(�A=q>��?��
B��f                                    Bx��  �          A{��ffA�����L��B�ff��ffA ��?G�@�B���                                    Bx��-�  T          A���33@��=�Q�?�B�����33@�{?fff@�(�B���                                    Bx��<h  �          A{��ff@���ff�,(�B�  ��ff@�>�Q�@
�HB���                                    Bx��K  T          A(���p�AQ�u��33B���p�A�?L��@�{B�B�                                    Bx��Y�  �          A(���\)A=q>���?��
B�W
��\)A��?�
=@�p�B���                                    Bx��hZ  �          A(����RA
=?��
@�G�B�\)���RA(�@
=qAL��B�k�                                    Bx��w   T          A����{Ap�>�@+�B�����{A   ?��@�
=B�p�                                    Bx����  �          AQ���
=@�
=�����ffB�p���
=@��R?\)@R�\B��\                                    Bx���L  
�          A(���@�  ��G��(��B��f��@��R?8Q�@��B��                                    Bx����  �          A���p�@��\��\)��(�B�8R��p�@�G�?E�@���B�z�                                    Bx����  �          A(���G�@�33=�\)>��B�Ǯ��G�@���?n{@�ffB�(�                                    Bx���>  T          A������@��R�#�
�uC �����@�p�?W
=@��C 33                                    Bx����  �          A\)���@��>L��?�
=C �����@�?��@�z�C ��                                    Bx��݊  �          A���@�R?fff@��C � ���@�=q?�(�A%G�CO\                                    Bx���0  �          A��=q@��?Tz�@�CL���=q@��?�33AffC�{                                    Bx����  �          A�H���@�?^�R@�p�C �����@�33?��HA#\)C��                                    Bx��	|  �          A����@�G�?h��@��C ����@�z�?�G�A'
=CW
                                    Bx��"  �          A�\��33@�p�?c�
@�G�Cn��33@��?޸RA%C�                                    Bx��&�  �          A�H��ff@�\?��@�  C8R��ff@�(�@��ALz�C                                      Bx��5n  T          A��˅@�{?:�H@���C���˅@��?˅AG�C�                                    Bx��D  T          A����\)@���?5@�(�C�)��\)@�z�?�=qA��C)                                    Bx��R�  �          AQ��ƸR@��?5@���C���ƸR@�(�?˅A��C\                                    Bx��a`  �          A  ��p�@��?(��@xQ�CL���p�@�?\A�C�\                                    Bx��p  �          A  �˅@�\?z�H@�G�C���˅@��?���A/33C�{                                    Bx��~�  
�          Aff�ʏ\@�ff?���@�z�CW
�ʏ\@��?�
=A8��C�                                    Bx���R  �          A�R��33@���?�A
=C����33@�@�AYCz�                                    Bx����  �          A=q��G�@�  ?c�
@���C���G�@��H?�\A(��C��                                    Bx����  �          A�ƸR@��>��H@:�HCaH�ƸR@�{?���A(�C�{                                    Bx���D  �          A=q��
=@��?Q�@�(�Ch���
=@���?��HA$  C�                                    Bx����  �          A���ȣ�@�p�?\)@XQ�C(��ȣ�@�G�?���A�C��                                    Bx��֐  �          A���ff@�Q�?@  @�  CxR��ff@�?�33A�HC\                                    Bx���6  �          A����
=@�\)?���@���C����
=@���@�AC�C��                                    Bx����  �          A���33@陚?E�@�(�Cٚ��33@���?�Q�A"�HCu�                                    Bx���  �          A\)���@�33>\@z�B��H���@�?���A�HB��                                    Bx��(  �          A(����@�p�?�R@o\)C �\���@���?ǮA�C�                                    Bx���  �          A  ��Q�@��?@  @��Cu���Q�@���?�Q�A$  C�                                    Bx��.t  
�          A���=q@��?Y��@�(�C  ��=q@�33?�  A*=qC�3                                    Bx��=  
�          A������@�
=?&ff@{�C L�����@�=q?�\)Az�C ޸                                    Bx��K�  �          A(����@�?��@Tz�B�z����@�p�?��A��B��                                     Bx��Zf  �          A\)���
@���    �#�
B�8R���
@��R?��\@�
=B��                                    Bx��i  �          A�
��@���    ���
B���@��R?��
@�\)B�=q                                    Bx��w�  �          A�
��=q@�>�z�?�(�B��R��=q@�(�?��
@�Q�C B�                                    Bx���X  �          A�\��ff@�
==��
?�B�����ff@�z�?��@�z�B�=q                                    Bx����  �          A����p�@��H�#�
���B�L���p�@���?�G�@ÅB�                                    Bx����  �          A�
���R@�  >aG�?��B�L����R@���?��\@�p�B�                                      Bx���J  �          A
=���
@�  <#�
=#�
B�aH���
@�?��@�\)B��H                                    Bx����  �          A
=���@��
���E�B�33���@�=q?k�@�33B���                                    Bx��ϖ  �          A�R��
=@��=#�
>k�B���
=@��H?��@�p�B��\                                    Bx���<  �          A
=��{@�
=���O\)B�L���{@��?p��@�ffB��3                                    Bx����  T          A����@�(������G�B�����@�?=p�@���B��H                                    Bx����  T          A����G�@�
=��33�
=qB�.��G�@�{?333@���B�\)                                    Bx��
.  �          A���@��R���:�HB�����@���?xQ�@�B��=                                    Bx���  �          A����(�@�z᾽p���B�=q��(�@��
?5@�(�B�k�                                    Bx��'z  �          AG����@���Ǯ���B�����@�(�?333@��\B��                                    Bx��6   �          A������@������p�B��R����@��H?0��@�  B��)                                    Bx��D�  �          A�����
@���\�z�B�\)���
@�z�?0��@���B��=                                    Bx��Sl  �          Az�����@��H�����=qB������@陚?O\)@�  B�k�                                    Bx��b  �          A�H��Q�@�=q�
=q�Q�B��
��Q�@�=q?��@j=qB��H                                    Bx��p�  �          Aff����@�\)�z��c�
B��H����@��?��@W�B��)                                    Bx��^  �          A�����@��R��33���B�
=����@�?J=q@��B�B�                                    Bx���  �          A���=q@������!�B�����=q@��H?@  @�33B�                                    Bx����  
�          AG���  A�׿0�����B��  A��?��@R�\B�\                                    Bx���P  �          A�����HA�ÿ�\)��
=B��)���HA=q>�?J=qB�k�                                    Bx����  �          A�H���@�G�>��
?�B����@��?�Q�A��B��R                                    Bx��Ȝ  �          A{��G�@��
>��@�RC 0���G�@�
=?�G�A(�C �q                                    Bx���B  �          A�R���@�?#�
@z=qC&f���@�  ?�(�A(Q�C޸                                    Bx����  �          Aff��(�@�33>��H@?\)C�\��(�@�ff?ǮA�Cp�                                    Bx���  �          A�H���H@�?��@i��CT{���H@�  ?�Q�A$��C�                                    Bx��4  �          A�R���@��
?
=q@Q�C }q���@�ff?�z�A"ffC#�                                    Bx���  �          Aff���@�p�>���@�C8R���@��?�  A�RC��                                    Bx�� �  �          A�R��p�@�z�>��H@@  C����p�@�\)?�ffA�C�f                                    Bx��/&  �          A
=�ƸR@��H?   @C33C=q�ƸR@�p�?���A�
C��                                    Bx��=�  �          A\)��(�@�p�?&ff@~{C����(�@�\)?�G�A,(�CO\                                    Bx��Lr  T          A33�ƸR@�\?@  @���CE�ƸR@�(�?���A4Q�C�                                    Bx��[  �          A�H���@��H?c�
@�C&f���@��
?��HA?�
C�                                    Bx��i�  �          A=q��  @��H?��A Q�C
�q��  @���@  A^�RCT{                                    Bx��xd  �          Aff����@���?�(�@�RC	������@�Q�@p�AX��C
�3                                    Bx���
  �          A=q��
=@θR?�@�(�C#���
=@�{@�AV�HC	\)                                    Bx����  �          A=q��Q�@˅?���A33C����Q�@��@Q�Ak33C
�                                    Bx���V  �          A=q��@�
=?�G�@�\)C����@�@�AeC��                                    Bx����  �          A�R���@�G�?�ff@��
C0����@���@
�HAU�CE                                    Bx����  �          A�\��@ٙ�?�  @�(�CaH��@љ�@AMCxR                                    Bx���H  �          A��ҏ\@��H?�@���C\�ҏ\@��@�A[�C	T{                                    Bx����  �          A=q��
=@�\)?s33@���C	@ ��
=@��?�z�AG�C
ff                                    Bx���  �          A����@��?�@l��CE���@�z�?ǮA/�
C�                                    Bx���:  T          Aff�q�@߮��=q��\)B�Ǯ�q�@�=q=�Q�?#�
B�8R                                    Bx��
�  �          A�\��@�  �R�\��z�B�ff��@�ff�   �^�HB�8R                                    Bx���  �          Ap���{@�Q���33�  B��Ϳ�{@�z��J�H����B��f                                    Bx��(,  �          AG���(�@������
=B�ff��(�@�ff�Mp���=qBƮ                                    Bx��6�  �          A{�333@�
=�O\)��\)B�W
�333@�  >��@A�B�(�                                    Bx��Ex  �          A ���p��@�\)�z�H�ᙚB���p��@�G�>W
=?�  B�33                                    Bx��T  �          @�
=�
=@��#33��=qB��
=@�{����Q�B�=q                                    Bx��b�  �          @��H��@�Q��n{��Q�B�aH��@ᙚ�#�
����B��
                                    Bx��qj  �          @�ff����@�z��U��z�BƊ=����@����~{B�aH                                    Bx���  �          A=q�5@�
=�C�
��33B�녿5@�z��p��E��B�W
                                    Bx����  �          A(����@�\�>�R��\)BǊ=���@�\)�У��6�HBƔ{                                    Bx���\  �          A{��z�@�p��U��G�B�\��z�@�z���
�k�B̅                                    Bx���  T          @���   @�{���
�Qz�B�8R�   @����R�)��B��                                    Bx����  T          Az���@׮����Bͮ���@��Tz���\)B�\)                                    Bx���N  T          Ap��\)@�\�Dz����HBٙ��\)@�  �޸R�Ap�B׮                                    Bx����  T          A  �'
=@��E��Q�B����'
=@�����H��B��H                                    Bx���  �          A���@  @����HQ���
=B�  �@  @��H����T��B�u�                                    Bx���@  �          Aff��
@�=q�u��Q�B�8R��
@�z��!����HBճ3                                    Bx���  �          A33��ff@��`����p�B�p���ff@������k33B�                                      Bx���  �          A��0��@��R�P  ��ffB�.�0��A�R��=q�I�B���                                    Bx��!2  �          A33��Q�A
=�:�H��33B�Lͽ�Q�A	G����ffB�=q                                    Bx��/�  �          AQ�>�A  ���b=qB�B�>�A�
���vffB�L�                                    Bx��>~  �          A�R��p�A������RB�=q��p�A���Y������B�\                                    Bx��M$  �          A�R�p��@��H�5��(�B��R�p��A�������B�{                                    Bx��[�  �          A�\��p�@���AG���{B�z��p�AG���=q�-��B�33                                    Bx��jp  T          A	����\@���5��z�B�{���\A{����  B�B�                                    Bx��y  �          A�ÿ�
=@�ff�-p�����B���
=A�ÿ�(��
=B��                                    Bx����  �          A(�����A  ��
�u�B��H����A�׿E�����B�{                                    Bx���b  T          A
{���Aff����l��B�Ǯ���A�R�.{��  B�                                    Bx���  �          A
ff��\@����C�
���HB�aH��\A\)��=q�)B��f                                    Bx����  �          A�����A��@������B�ff����A�
��(���RB�k�                                    Bx���T  �          A�׿�ffA����R��G�B�#׿�ffA	���h����{B�aH                                    Bx����  �          A
=�n{A33�(Q����B�녿n{A�׿�����\B�ff                                    Bx��ߠ  �          A=q�s33@�
=�s�
��G�B��{�s33A������l(�B��                                    Bx���F  �          AQ�p��A��dz�����B�
=�p��A�
��p��J�RB�B�                                    Bx����  �          A����@�(���=q��Q�B�G����A���1G����RB��                                    Bx���  �          AG���  @��������\)B�aH��  A	��0  ��p�B�B�                                    Bx��8  �          A  �
=A=q�u���  B�  �
=A\)���Z�RB��                                     Bx��(�  �          Ap��O\)@����|����{B��H�O\)Aff�=q�~�HB�
=                                    Bx��7�  �          A����H@���Vff���B�z���HA �ÿ�{�O�
B�                                    Bx��F*  
�          @���:�H@�  �%����B���:�H@���33���B���                                    Bx��T�  �          @���33@θR�Mp���BШ���33@�ff���p��Bγ3                                    Bx��cv  �          @����(�@�(��HQ���z�B�  ��(�@�33��G��Q��B�B�                                    Bx��r  �          @�z�� ��@ڏ\�W��ʸRB�Ǯ� ��@�33�   �k33BҔ{                                    Bx����  �          @��R�z�@ʏ\�����B�#��z�@�G��@�����\BԳ3                                    Bx���h  �          A���\)@�p��hQ���G�B�LͿ�\)@��G���\)B��                                    Bx���  �          A�
���\A	G���{�
=B��{���\A
�H>���@Q�B�p�                                    Bx����  �          A��\A�Ϳ�33�.�\B���\A\)=#�
>�=qB�
=                                    Bx���Z  �          A33�!G�A(���\)��{B��3�!G�A�?�\@_\)B���                                    Bx���   �          A33�5@�����jffB�LͿ5A�\���H�\��B���                                    Bx��ئ  �          A�;�A{����ffB��)��A�>��R@Q�B��
                                    Bx���L  T          A�\@�ff�7����B�  �\@��
�����p�B���                                    Bx����  �          A�����@�Q�������Q�B�z����@���#�
���B��                                    Bx���  �          A33��@����   �eB�Ǯ��A �׾�
=�AG�B��=                                    Bx��>  �          A�=��
@�z��Q����B��R=��
@�z��G��H��B���                                    Bx��!�  �          A�\@G�@����  ���\B�#�@G�@��H�O\)��=qB��                                    Bx��0�  �          A�
?G�@�(��У��:ffB��q?G�A �ͼ��uB���                                    Bx��?0  �          AG��&ff@�\�333���HB�L��&ff@�  ��=q�z�B�8R                                    Bx��M�  �          A �����@�\�6ff���RBճ3���@�Q쿰��� Q�B��H                                    Bx��\|  �          @�p��#33@�(��/\)��{Bܨ��#33@�G������Bڊ=                                    Bx��k"  �          @����ff@�=q����ffB����ff@ƸR��\)��\B�B�                                    Bx��y�  �          @�\)�P  @�ff����Q�B���P  @�p���33�#�
B�Ǯ                                    Bx���n  �          A�R��33@��Ϳ�=q�R�HB�G���33@��
��  ���
Bˣ�                                    Bx���  �          A ���s�
@�
=���\���B�W
�s�
@�\>#�
?�B�{                                    Bx����  �          A Q���(�@У���\�l��B��3��(�@ٙ��+���=qB�\                                    Bx���`  �          @�
=��  @�\)�Q���Q�B����  @ȣ�>�@VffB�\)                                    Bx���  �          @�
=���@�33��z��#33B�z����@�  �#�
��  B�W
                                    Bx��Ѭ  �          @�p���Q�@�G����H�G�B�#���Q�@���>��?��B�B�                                    Bx���R  �          @�\)�g
=@޸R���H�)G�B��g
=@��#�
�#�
B�q                                    Bx����  �          @�{�XQ�@�\)��\)�=�B�q�XQ�@�������\)B�{                                    Bx����  �          @��z=q@�\)���R�-B�W
�z=q@��ͽ��
���B�(�                                    Bx��D  �          @�����  @�G����R��B����  @�
=?}p�@�  B��R                                    Bx���  �          @��Vff@ҏ\������B����Vff@љ�?Y��@ӅB�33                                    Bx��)�  �          @��@��@�p������B�=�@��@�Q�>u?��B��                                    Bx��86  �          @�33�k�@�녿@  ���\B�{�k�@�=q?z�@�G�B�p�                                    Bx��F�  �          @�p��fff@�p��c�
�ۅB�fff@θR>��@h��B�G�                                    Bx��U�  �          @�ff�j�H@θR���H�o\)B�=q�j�H@�p�?^�R@׮B�\                                    Bx��d(  �          @�\)�p��@�ff�J=q��z�B�p��@ָR?!G�@���B�                                     Bx��r�  �          A ����z�@ڏ\�}p���33B�\)��z�@�z�>�@Q�B��                                    Bx���t  �          A����p�@ָR�����(�B�����p�@���>Ǯ@1�B�#�                                    Bx���  �          @����Q�@љ��
=��\)B����Q�@���?O\)@�G�B�Q�                                    Bx����  �          @��H��(�@��
�#�
��Q�C �\��(�@�Q�?���A�C
=                                    Bx���f  �          @�
=���R@Ǯ�\�4z�B�(����R@�?s33@�B��                                    Bx���  �          @���z�@ȣ׿(����ffB�
=��z�@ȣ�?.{@��\B�\                                    Bx��ʲ  �          @��
���@ƸR�.{����B�.���@ƸR?&ff@���B�(�                                    Bx���X  �          @��
�j�H@ڏ\��33��\B�p��j�H@�p�>�{@��B���                                    Bx����  �          @�(��
=@�(��#33��\)B��
=@�Q�xQ����HB�p�                                    Bx����  �          @�{���@�=q�U��ffB����@��
��ff�Yp�B�p�                                    Bx��J  �          @�\��@�Q��vff��Q�B�#׿�@�������B�G�                                    Bx���  �          @�(����R@߮�3�
���RB�\���R@����R��B��f                                    Bx��"�  T          A (���G�@��7���z�B�\)��G�@�녿�(����B�B�                                    Bx��1<  �          AG���{@�p��3�
����B���{@�33��33�33Bǣ�                                    Bx��?�  �          A����\)@�(����}��B�Q��\)A�H���hQ�B�u�                                    Bx��N�  �          A  ��G�@�{�����\)B͸R��G�A z�.{��\)B̨�                                    Bx��].  �          Aff���@���ff��Bͮ���A
=�
=��  B̳3                                    Bx��k�  �          A=q��33@�����B�aH��33A{�
=����B�Q�                                    Bx��zz  �          Az���@��R��H����BΔ{���A�Ϳ!G����RB͏\                                    Bx�   �          Ap���z�@�G�����Bπ ��z�A�z���  B�p�                                    Bx��  
�          A�\�G�@������V=qBЏ\�G�A�\�.{��33B�Ǯ                                    Bx�¦l  �          A����p�@�
=���
�E�B̀ ��p�A�R<#�
=#�
B��                                    Bx�µ  
�          A���HA �Ϳ�����B����HA�R>�33@�HB˙�                                    Bx��ø  �          A(��8��@�녿s33��
=B��H�8��@�\?:�H@�z�B�                                    Bx���^  T          @��R���@��
?�R@�{B�W
���@\@   Ah��B��                                    Bx���  	�          @�
=���R@˅?n{@�p�B����R@�  @33A�z�C �                                    Bx���  T          @�ff��z�@��H?��HAz�B�aH��z�@�p�@%�A��C 
                                    Bx���P  �          @�����@�?��A��B�\���@�\)@*�HA���C�                                    Bx���  �          @����\)@�=q>���@�RC�=��\)@�33?�{ABffC��                                    Bx���  "          @�p����\@ƸR?Q�@�p�B��R���\@�(�@
�HA33Ck�                                    Bx��*B  
�          @���z�@�  ?E�@��\C����z�@�@As�C+�                                    Bx��8�  T          @�
=����@��\?�G�A=qC�3����@��@{A��C�                                    Bx��G�  �          @�{��@�
=?�A  C ���@��@{A�(�C�                                    Bx��V4  T          @��H��33@�{?�G�AG�Ch���33@���@   A�{C��                                    Bx��d�  
�          @�z���(�@�\)?��A'�C!H��(�@���@%�A��
C��                                    Bx��s�  T          @�z����@�
=?У�AD��C�����@�
=@0  A�\)C
\)                                    Bx�Â&  "          @�33���@���?�33AB=qC33���@���@:=qA��C޸                                    Bx�Ð�  
�          @�(���
=@�z�@:�HA���Cff��
=@��@�  A�C!H                                    Bx�ßr  T          @�{���
@���@UA�33C
5����
@z�H@��HBp�C�
                                    Bx�î  �          A ����G�@��R@E�A�{C
(���G�@�z�@��
A�Q�C=q                                    Bx�ü�  �          @�=q����@�ff@QG�A�  C(�����@e@�{BC�                                    Bx���d  T          @����G�@z=q@G�A��\C�3��G�@Fff@z�HA���C�=                                    Bx���
  
�          @�\)����@���@8Q�A��C������@q�@u�A�(�C��                                    Bx���  �          @����R@���@^�RA��C�)���R@HQ�@���B	Q�C+�                                    Bx���V  �          @������@�=q@c33A��C}q����@I��@�(�B�\C:�                                    Bx���  
�          @��
���R@��@eA�
=C�{���R@S�
@��RBp�C�                                    Bx���  �          @�z����@���@eA�{C���@N{@�{B{C                                    Bx��#H  �          @����p�@|��@^{A���C����p�@C33@���B
{C�
                                    Bx��1�  �          @����z�@_\)@FffA�
=C^���z�@,(�@s�
A�(�Cz�                                    Bx��@�  �          @���ff@q�@Mp�A�33Cu���ff@<(�@\)A�
=C�                                    Bx��O:  �          @����(�@��R@N�RA˙�C��(�@vff@�
=BC�                                    Bx��]�  
�          @�=q���R@�=q@B�\A�p�C
�q���R@p  @�Q�A�C=q                                    Bx��l�  �          @�Q�����@���@>�RA�G�C������@^{@xQ�A�33CB�                                    Bx��{,  �          @������@�{@;�A��C
=����@h��@w�A�z�Cu�                                    Bx�ĉ�  �          @���
=@��@9��A�
=C����
=@��@|(�A�=qC��                                    Bx�Ęx  �          @�����@�Q�@P  AͮC!H����@xQ�@�Q�B
��C�                                    Bx�ħ  �          @�ff��\)@�(�@W
=A֣�C
����\)@_\)@���Bp�C                                      Bx�ĵ�  T          @�����(�@�z�@VffA�33C���(�@P��@��RB
�C�{                                    Bx���j  �          @�����@��
@N�RA�G�C&f���@P��@��HB  Cz�                                    Bx���  �          @����@��@EA��C
����@aG�@�Q�B
=CxR                                    Bx���  "          @�=q����@�z�@I��A�(�C)����@R�\@���B
=Cn                                    Bx���\  �          @����=q@��H@C33A�  C�f��=q@QG�@z�HB��C��                                    Bx���  �          @�������@��@C�
A���Ck�����@R�\@{�B=qC�
                                    Bx���  �          @�  ��=q@���@Tz�A��C	�R��=q@QG�@�{B�Ch�                                    Bx��N  T          @޸R��ff@���@J�HA�C.��ff@Z=q@��\B�C��                                    Bx��*�  �          @�  ���@�
=@G
=AӮC	z����@XQ�@�Q�B=qC��                                    Bx��9�  �          @����~�R@��@g
=A�Q�C���~�R@aG�@��B"z�C
��                                    Bx��H@  "          @�
=�{�@�
=@g�A�(�CT{�{�@`  @��B#��C
O\                                    Bx��V�  �          @�  �g
=@��\@vffBC B��g
=@c33@�=qB.33Cz�                                    Bx��e�  �          @ᙚ�Mp�@��\@�G�B�B���Mp�@\(�@��B@33C�q                                    Bx��t2  �          @�p��e�@�
=@n�RA��\B�p��e�@}p�@�G�B'��C{                                    Bx�ł�  �          @�z���33@��@QG�A�
=Cn��33@z=q@��B�C\)                                    Bx�ő~  T          @�\�S�
@���@|��BB�k��S�
@n{@��RB3��C�                                    Bx�Š$  �          @�33� ��@~{@�{BJ��B��� ��@%�@�
=By(�B���                                    Bx�Ů�  �          @��Ϳ˅@c33@�p�B^33B�\)�˅@
=@��HB��3B�                                      Bx�Žp  �          @ۅ��p�@��H@�z�BBG�B癚��p�@0��@��RBq\)B�.                                    Bx���  T          @��
�=q@�z�@��
B6{B�W
�=q@7�@��RBb��C�3                                    Bx��ڼ  �          @�(��2�\@�{@�{B(�B�p��2�\@dz�@�BCffC                                       Bx���b  �          @ڏ\�.{@��R@u�B\)B��.{@z�H@���B6�B���                                    Bx���  �          @�
=�3�
@�  @j�HA��B�\�3�
@�\)@��
B*��B�(�                                    Bx���  Y          @�녿�@z�H@x��B(�B�z��@8��@��RBXp�B�
=                                    Bx��T  �          @�
=�ff@o\)@���B3ffB�ff@'�@���Ba{C ��                                    Bx��#�  �          @�
=�0��@��\@,��A�Q�B����0��@���@u�Bz�B�\)                                    Bx��2�  �          @�  ��p�@���@�33B$�B�녿�p�@j�H@�(�BU��B왚                                    Bx��AF  �          @�Q��z�@�p�@�  B��B�G��z�@}p�@�33BJ��B�R                                    Bx��O�  �          @���(��@��@���Bz�B�ff�(��@��
@��B6�B�Q�                                    Bx��^�  �          @�Q��$z�@�33@�p�Bp�B�\)�$z�@�ff@�=qB<�HB��                                    Bx��m8  T          @���#�
@�ff@w�B��B�W
�#�
@��
@��B3B�R                                    Bx��{�  �          @�G��8��@���@s33A��B�Q��8��@��R@�Q�B-=qB���                                    Bx�Ɗ�  �          @�\)�aG�@��\@8��A�33B��aG�@�\)@�(�B(�B�W
                                    Bx�ƙ*  Y          @�z��P  @���@'
=A��
B��
�P  @�\)@z=qB(�B�L�                                    Bx�Ƨ�  
�          @���R@�33?�{A�B�����R@�  @E�A���B�.                                    Bx�ƶv  �          @�  �
=@��?�Q�A���B���
=@~{@>{B33B�k�                                    Bx���  �          @�G���H@��@Q�A�\)B���H@|��@[�B�HB�\                                    Bx����  T          @��\��(�@���@�A��B����(�@�(�@J=qB
z�B��f                                    Bx���h  �          @��׿�{@�\)@�A�\)B��Ϳ�{@��H@L(�BB���                                    Bx���  �          @�=q��R@�
=@	��A�Q�B���R@u�@J=qB
B��H                                    Bx����            @��
�*�H@�\)?�=qA�
=B��{�*�H@j�H@2�\A���B�.                                    Bx��Z  Y          @�����@�\)?�=qA���B�8R���@n�R@"�\A��B���                                    Bx��   �          @��Mp�@��\@Q�A�  B����Mp�@�p�@Q�A�(�B�                                      Bx��+�  �          @�\)����@�G�?�{A\��Ck�����@�Q�@1G�A\C�
                                    Bx��:L  �          @�����z�@��\?�=qAS�
C����z�@���@0  A�=qC�3                                    Bx��H�  �          @������@�33?�Q�A<��B�Ǯ����@��H@0  A�33C ��                                    Bx��W�  �          @ָR�U�@��R?��A\)B�ff�U�@���@��A�
=B�B�                                    Bx��f>  �          @Ǯ�J=q@��R?\(�A ��B� �J=q@�33@A���B�33                                    Bx��t�  	3          @�33�'�@��
?(�@ȣ�B��'�@��\?�G�A��B�                                    Bx�ǃ�  	�          @�  �*�H@��R?z�@�(�B��)�*�H@�{?��A�  B�{                                    Bx�ǒ0  �          @�Q��{@�p�>���@UB����{@�{?���A~{B�\                                    Bx�Ǡ�  �          @�  �   @�=q�#�
��{B؊=�   @�G�?Y��@��Bخ                                    Bx�ǯ|  �          @�(��J=q@�33���uB�q�J=q@�?�Q�AA�B��                                    Bx�Ǿ"  �          @ָR�L��@��
<�>�\)B�=q�L��@�{?���AH(�B�                                    Bx����  �          @�p��K�@�(�>aG�?�{B����K�@���?ٙ�Ac�B��                                    Bx���n  �          @�
=�Y��@\=�\)?�B�ff�Y��@�(�?��AK�B�                                    Bx���  �          @޸R�o\)@��<#�
=uB�(��o\)@�?�A<��B��q                                    Bx����  �          @�{�p  @��H=�G�?h��B�k��p  @�z�?\AJ=qB�=q                                    Bx��`  �          @����Z�H@�
=��z���B�Z�H@��?�z�A33B��                                    Bx��  �          @��H�Tz�@�p�����z�B�Tz�@��
?n{@�=qB�                                    Bx��$�  �          @������@�����
�&ffB�{����@��H?��A,z�B��                                    Bx��3R  �          @�=q�W
=@�녿�p��v{B����W
=@�=q�Ǯ�\��B�.                                    Bx��A�  �          @��*=q@�(���
=�FffB��*=q@��ü��
�B�\B�8R                                    Bx��P�  �          @�z��*=q@�p������N{B�aH�*=q@�33?W
=A33B�{                                    Bx��_D  �          @�G��7
=@���W
=�
=B����7
=@�\)>��
@S�
B�L�                                    Bx��m�  �          @����7
=@���������B��{�7
=@�{�����{B�3                                    Bx��|�  �          @�(�� ��@Tz��8Q��(�B�aH� ��@z�H���R��(�B�Q�                                    Bx�ȋ6  T          @�Q��<(�@@��@)��B �HCY��<(�@G�@S33B&ffCE                                    Bx�ș�  �          @�
=�S33@C33@EB
�C	L��S33@p�@o\)B-{C@                                     Bx�Ȩ�  �          @��R�C�
@HQ�@dz�B�RCW
�C�
@
�H@�
=BAz�C��                                    Bx�ȷ(  �          @����HQ�@P  @hQ�BQ�C�3�HQ�@G�@��B@p�C\                                    Bx����  �          @��h��@e@@��A�{Cn�h��@0  @r�\B��C�3                                    Bx���t  �          @��\�C�
@Z�H@g
=B�CǮ�C�
@(�@��\B?�C\)                                    Bx���  �          @�33���@0  @�  BQ�B�(����?У�@�G�B|��C	�H                                    Bx����  "          @���33@;�@��HBT�
B�(���33?��@�B�u�C )                                    Bx�� f  �          @��
���
@>�R@�G�BS�\B♚���
?���@�z�B��B��                                     Bx��  �          @�녿���@9��@��BM�HB��H����?�ff@��B|�C��                                    Bx���  �          @�G��(�@h��@2�\B(�B�\�(�@6ff@eB3��B�                                    Bx��,X  �          @�G���G�@r�\@2�\B{B��Ϳ�G�@@  @hQ�B9  B���                                    Bx��:�  T          @�Q쿢�\@x��@8��B��B�8R���\@Dz�@p  BA��B�                                    Bx��I�  �          @�zῃ�
@��H@<��B�B�33���
@P  @vffBA�
B�\                                    Bx��XJ  T          @�=q�J=q@��@Q�Bp�Bɨ��J=q@L(�@�BOp�Bϳ3                                    Bx��f�  �          @��Ϳ\)@�G�@a�B#ffBã׿\)@C�
@��BZ��BȊ=                                    Bx��u�  �          @�33���@W�@\��B5{B�p����@�@��BlQ�B�W
                                    Bx�Ʉ<  �          @�Q��@�ff@L(�B�B\��@b�\@�BEffB�=q                                    Bx�ɒ�  �          @�(��aG�@���@G
=B��B�W
�aG�@Y��@��BD��B�.                                    Bx�ɡ�  �          @�G��Tz�@�{@-p�A�G�B�G��Tz�@i��@mp�B2�RBͽq                                    Bx�ɰ.  �          @�G���33?�Q�@��A��HC#�\��33?#�
@
=A�p�C+33                                    Bx�ɾ�  �          @�  ��
=?��@Aޣ�C"c���
=?:�H@&ffA��C*.                                    Bx���z  �          @�  ����?�33@9��A��
C$�=����?&ff@J=qA�=qC,��                                    Bx���   �          @�(���\)>.{@)��A���C2+���\)��@'
=A�\)C9B�                                    Bx����  �          @�33���?p��@0  AָRC*
���>u@9��A�
=C1p�                                    Bx���l  �          @�p���
=?c�
@1G�A�=qC+!H��
=>B�\@9��A���C2\                                    Bx��  T          @�\)���
?��\@#33A��C*#����
>�p�@.{A�ffC0c�                                    Bx���  �          @������
?�  @$z�A��
C'�R���
?
=@333A�\)C.J=                                    Bx��%^  �          @�  ���?�Q�@ ��A�ffC(xR���?��@.�RA�
=C.�f                                    Bx��4  �          @��H��p�?�G�@'
=A�Q�C(  ��p�?
=@5�AˮC.T{                                    Bx��B�  �          @������H?���@�
A��\C'���H?8Q�@#�
A�G�C-=q                                    Bx��QP  �          @ҏ\��(�?�Q�@�\A�z�C)
=��(�?(��@��A�G�C-�H                                    Bx��_�  �          @Ϯ��z�?n{?��A�
=C+k���z�>�G�@33A���C/��                                    Bx��n�  �          @����G�?L��@G�A�
=C,�=��G�>�\)@	��A��RC1n                                    Bx��}B  �          @�p���G�?8Q�@A�{C-� ��G�>B�\@��A�(�C2J=                                    Bx�ʋ�  �          @������?   @
=A��C/=q���ü�@
�HA�ffC4@                                     Bx�ʚ�  
�          @�����33?z�?��
A��C.����33>�?�\)A�(�C2�                                     Bx�ʩ4  �          @�33��33?8Q�?���Ae��C-O\��33>��R?ٙ�Axz�C1\                                    Bx�ʷ�  "          @�ff��\)?G�?�p�AT��C,�H��\)>���?У�AiC0^�                                    Bx��ƀ  "          @�\)�Ǯ?Q�?��RAT��C,�=�Ǯ>�(�?��Ak�C0
=                                    Bx���&  �          @����\)?:�H?��A=G�C-J=��\)>Ǯ?���AQ��C0c�                                    Bx����  
Z          @�{���?!G�?��A  C.W
���>�33?�
=A)p�C0�{                                    Bx���r  T          @�\)����?!G�?��HAP  C.E����>��?���A`  C1��                                    Bx��  �          @�\)��=q?z�?��\A4z�C.Ǯ��=q>�  ?�\)AC\)C1�3                                    Bx���  "          @������?   ?�\)AC/�����>aG�?��HA*ffC2\                                    Bx��d  
�          @�G����
?W
=?���A)C,s3���
?�?�\)AAC/B�                                    Bx��-
  �          @����Ϯ?Q�?�
=A#�C,�=�Ϯ?�\?��A:ffC/�                                     Bx��;�  T          @У�����?(��?��A{C.#�����>\?�Q�A(Q�C0�
                                    Bx��JV  
�          @У���
==�?8Q�@�33C2�3��
=�u?:�H@�p�C4�                                    Bx��X�  �          @�  ��ff=�Q�?G�@ۅC3(���ff����?G�@�33C4ٚ                                    Bx��g�  
�          @ȣ���{>k�?E�@�33C1�f��{<�?L��@�z�C3��                                    Bx��vH  �          @����z�>.{?�\@��C2p���z�=L��?
=q@��C3�
                                    Bx�˄�  �          @�{����>�Q�?!G�@�z�C0�����>B�\?333@�
=C2T{                                    Bx�˓�  �          @���z�>��
?��@��
C1+���z�>#�
?(��@�z�C2��                                    Bx�ˢ:  �          @�������?���>�\)@+�C)+�����?��
?��@�ffC*�                                    Bx�˰�  �          @�=q����?�(����Ϳ�ffC"�����?�Q�>��R@O\)C"G�                                    Bx�˿�  �          @Ǯ��  �k�?Y��@��RC<� ��  ����?��@�(�C>ff                                    Bx���,  
�          @�  ���
�Y��?^�RA ��C;����
���?#�
@���C=��                                    Bx����  �          @��H���þ�p�?G�@���C7aH���ÿ��?(��@�Q�C8��                                    Bx���x  
�          @�\)��{�L��?�R@���C5���{��{?\)@�ffC7�                                    Bx���  
�          @����Ϯ=��
?+�@���C3@ �Ϯ���
?+�@��C4�3                                    Bx���  T          @������>��>��H@���C1�3����>\)?
=q@�
=C2��                                    Bx��j  T          @��
�Ӆ=�Q�>�Q�@FffC3B��Ӆ    >�p�@L(�C4                                    Bx��&  "          @Ϯ�Ϯ=�\)>B�\?�33C3\)�Ϯ<�>L��?�G�C3Ǯ                                    Bx��4�  
�          @�33�ʏ\��  >�p�@VffC6@ �ʏ\����>���@-p�C6�q                                    Bx��C\  �          @���˅�   ?�R@���C8s3�˅�!G�>�@���C9�f                                    Bx��R  �          @�����
��>��H@��C8����
��R>�33@J�HC9��                                    Bx��`�  T          @�
=�θR��Q�>��?��C7+��θR�\=u?
=qC7ff                                    Bx��oN  
�          @Ϯ�θR��>��@G�C8�)�θR��=�?���C9�                                    Bx��}�  �          @�ff��p��   >aG�?���C8u���p��
=q=�Q�?Tz�C8��                                    Bx�̌�  �          @�����
�   >�{@A�C8z����
��>W
=?��C9�                                    Bx�̛@  �          @�
=��ff���aG�� ��C8:���ff�Ǯ���
�=p�C7�H                                    Bx�̩�  "          @Ǯ�ƸR�Ǯ=�G�?��C7�\�ƸR����<��
>#�
C7��                                    Bx�̸�  "          @�
=��{��׽#�
����C8T{��{��G������C8)                                    Bx���2  T          @�����(���33<��
>B�\C7@ ��(���{�u�
=C733                                    Bx����  T          @�{��p����R����z�C6޸��p���z���ͿuC6�R                                    Bx���~  �          @Å���H��p�������C7}q���H���þW
=�   C7)                                    Bx���$  �          @�Q����R����!G�����C6}q���R���Ϳ+���ffC4��                                    Bx���  �          @�=q��Q쾳33��R��z�C7L���Q�B�\�.{��  C5��                                    Bx��p  �          @������B�\��{�P��C;33����&ff��\���
C:8R                                    Bx��  �          @�\)���R��׾����RC8}q���R�Ǯ��Q��^{C7Ǯ                                    Bx��-�  �          @�����\)�+��#�
����C:p���\)�   �G���(�C8�{                                    Bx��<b  "          @��������R�&ff��C6���������333�ָRC5n                                    Bx��K  �          @��R��{�\���R�@  C7�H��{��z�Ǯ�qG�C6��                                    Bx��Y�  �          @�p����;�  ��p��hQ�C6ff���;����
=���HC5z�                                    Bx��hT  
�          @�33���H��  >#�
?���C6z����H��\)=���?xQ�C6�=                                    Bx��v�  �          @�=q���ÿ��=�G�?��C9�����ÿ���#�
�\C9�R                                    Bx�ͅ�  "          @��\�������;B�\��C7�������{��\)�-p�C7^�                                    Bx�͔F  
(          @���G���{����  C>����G��xQ�5��C=�=                                    Bx�͢�  �          @����  ��
=�&ff�ÅC?&f��  ��  �h�����C=k�                                    Bx�ͱ�  �          @�G���zῗ
=�(���Q�C?T{��z῁G��^�R�(�C=�                                    Bx���8  T          @ƸR��33��  ��G����C=Q���33�^�R�(����(�C<)                                    Bx����  
�          @���\�fff������HC<h��\�B�\�+���{C;+�                                    Bx��݄  	�          @��\���׿������%G�CC�����׿�  �����Z�RC@                                    Bx���*  
�          @��H���
���ÿO\)��ffCAE���
���Ϳ���-�C?�                                    Bx����  Z          @��\����333�aG���C:������!G��\�p  C:B�                                    Bx��	v  �          @�33��Q�k��Ǯ�hQ�C<����Q�L�Ϳ
=����C;��                                    Bx��  
�          @�����
=�Y����z��-p�C<!H��
=�B�\������C;G�                                    Bx��&�  Z          @Å��=q�����33�S�
C9���=q�   ����G�C8�                                     Bx��5h  N          @�z��Å�
=���
�5C9� �Å��;W
=��(�C9.                                    Bx��D  �          @�����k�=�Q�?J=qC6(������  <�>�=qC6J=                                    Bx��R�  
�          @����(�>W
=>�@��\C2���(�=�Q�?   @��C3)                                    Bx��aZ  �          @�  ��>�(�?E�@�C0  ��>u?Y��@��HC1�                                    