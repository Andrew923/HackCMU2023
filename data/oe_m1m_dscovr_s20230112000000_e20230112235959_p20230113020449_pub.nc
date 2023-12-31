CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230112000000_e20230112235959_p20230113020449_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-01-13T02:04:49.454Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-01-12T00:00:00.000Z   time_coverage_end         2023-01-12T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill         �   records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxZ4k�  
�          @�(�@��@Vff@�  B(�B(�@��?��@��
B6�A�Q�                                    BxZ4zf  "          @��@|��@��@�BG�BN��@|��@S33@���BA��Bff                                    BxZ4�  "          @���@��R@'�@�p�B�A���@��R?xQ�@���B){A�                                    BxZ4��  	�          @��R@��@@��@��B	��A���@��?�33@�
=B$�\AXz�                                    BxZ4�X  �          @��R@�ff@9��@�{B�HA�  @�ff?��R@��
B*��AC�
                                    BxZ4��  
�          A ��@��R@c�
@��A�RA���@��R@�
@�  B�\A��R                                    BxZ4ä  �          A ��@�z�?�ff@���B�Abff@�zὣ�
@���B��C��q                                    BxZ4�J  �          A�R@��@<(�@�33A�G�AĸR@��?��H@��\B{AL��                                    BxZ4��  T          A�@��@X��@��\B�
A�R@��?�@�{B (�A��\                                    BxZ4�  "          A�@�G�@c33@��A���A�@�G�?�(�@�
=B�A��R                                    BxZ4�<  �          A�R@�Q�@:=q@��\B p�A�33@�Q�?�{@�G�B�AC�
                                    BxZ5�  T          A\)@�{@c�
@�G�A��
A�
=@�{@z�@��RB
=A��                                    BxZ5�  "          A(�@�p�@|(�@tz�A�ffBG�@�p�@ ��@�(�B  A���                                    BxZ5*.  T          AG�@�\)@���@aG�A��
B	��@�\)@:=q@�ffB	p�A�ff                                    BxZ58�  �          A�R@�ff@tz�@.{A���A�z�@�ff@0  @s�
A��A�z�                                    BxZ5Gz  "          A@�=q@��@z�HA�B"z�@�=q@C�
@�p�B ��A�                                    BxZ5V   �          A��@�ff@���@_\)A̸RB�R@�ff@J�H@�  BQ�A�ff                                    BxZ5d�  T          AG�@��@z=q@�B�BQ�@��@�@�ffB,G�A�ff                                    BxZ5sl  
�          A{@��@�z�@I��A���B�R@��@X��@�
=BG�A�=q                                    BxZ5�  
�          A{@�{@���@?\)A�(�B033@�{@���@���B�RB�                                    BxZ5��  "          Ap�@��@�ff@^{A��B*�@��@e�@��
B(�B
=                                    BxZ5�^  �          A@�p�@�z�@�G�A���B�
@�p�@7
=@�\)B!�HAޣ�                                    BxZ5�  
�          @���@��@��@l��Aߙ�B9��@��@g
=@��
B!�RBff                                    BxZ5��  
�          A ��@�@�z�@FffA�Q�BT33@�@��H@�=qB(�B8�                                    BxZ5�P  �          A ��@h��@޸R?�(�AF{Byp�@h��@���@vffA��Bk�                                    BxZ5��  �          AQ�@Z=q@�?���@�p�B���@Z=q@ָR@XQ�A��B|{                                    BxZ5�  
�          A ��@7
=@�>���@7
=B�(�@7
=@�
=@1G�A�G�B�\)                                    BxZ5�B  �          A�R@0��@��
?�@j=qB�.@0��@��@:�HA�z�B�G�                                    BxZ6�  �          A��@�
=@��
@���B (�BTp�@�
=@{�@��RB7��B+�H                                    BxZ6�  
�          A�@��@p��@���B�BG�@��?�@\B?\)A���                                    BxZ6#4  "          A   @��
���@��B.z�C�\)@��
�U@��\BG�C�p�                                    BxZ61�  �          A@/\)����@���B�C�y�@/\)��@)��A���C���                                    BxZ6@�  
�          @�p�?�(����H@���B(�C��
?�(���R@
=A�Q�C���                                    BxZ6O&  �          @���@_\)���@���B4�RC�@_\)����@y��A�{C���                                    BxZ6]�  �          @��\@�  ����@W
=A�Q�C�Z�@�  ��G�?���A (�C���                                    BxZ6lr  "          @�@8����p�@���B!�C��
@8����  @G�A�C��q                                    BxZ6{  �          @�@����
=@p��A�
=C�  @����\?\A3
=C�                                    BxZ6��  T          @�p�@����w
=@���B�C��{@�����p�@S33A���C��=                                    BxZ6�d  �          @��R@��\���@�33B��C���@��\����@:�HA���C�4{                                    BxZ6�
  
�          @�\)@E��{@@��A��C�^�@E�陚?=p�@�33C�s3                                    BxZ6��  �          AG�@+���33@N{A�
=C��@+����?c�
@�z�C��\                                    BxZ6�V  �          A�?��R��(�@=p�A���C���?��R��>�@\��C�t{                                    BxZ6��  "          A(�@�  �1�@��BQ�C�%@�  �|(�@A�A֣�C�n                                    BxZ6�  
Z          Ap�@�z��@�  B'=qC��H@�z��(�@�=qB��C�/\                                    BxZ6�H  T          @�ff@�����@���BB�C�<)@���J�H@�p�B$��C�Ǯ                                    BxZ6��  T          @��\@�z�\)@��B"�C�S3@�z��(�@�Q�BG�C���                                    BxZ7�  	�          @�  @�녾�@�=qB.�
C�W
@�녿��@�Q�B"p�C�{                                    BxZ7:  �          A ��@�(�@�Q�@�ffBQ�B�@�(�@�@�
=B5A��
                                    BxZ7*�  
�          Aff@��@Z=q@�ffB�A���@��?�\@���B/=qA��R                                    BxZ79�  
�          A�H@�
=@�@�p�BG�A�33@�
=?G�@��B{@��
                                    BxZ7H,  T          A ��@˅?=p�@�G�B
=@���@˅�J=q@���BC�t{                                    BxZ7V�  "          A z�@��
����@���BffC�&f@��
��H@n�RA���C���                                    BxZ7ex  
Z          A ��@ָR��=q@�  A��C�^�@ָR�?\)@N�RA��C�H                                    BxZ7t  "          @�z�@��H�fff@>�RA�  C�/\@��H��33?��HAH��C��                                     BxZ7��  "          @�(�@�z��hQ�@'
=A�Q�C�5�@�z���Q�?���A\)C�(�                                    BxZ7�j  "          A Q�@�33��@R�\AÅC�w
@�33�QG�@��A�  C�9�                                    BxZ7�  �          @�{@Ϯ�i��@!�A�=qC�S3@Ϯ��Q�?��\A��C�`                                     BxZ7��  "          @��@�\)���@.�RA��HC��@�\)��Q�?���@��\C�g�                                    BxZ7�\  "          @��@�����(�?�Q�AfffC��)@�����\)>�  ?�{C���                                    BxZ7�  "          @��
@�ff��Q�@:�HA��C��{@�ff��{?���A"=qC���                                    BxZ7ڨ  T          @��@������?�Q�A(Q�C��H@������׽��h��C�(�                                    BxZ7�N  "          @�z�@�G����@s�
A�C�ff@�G�����@�\A�ffC��                                    BxZ7��  �          @��@�p���\)@�RA��
C�Z�@�p���?
=@���C�!H                                    BxZ8�  �          @���@�p���(�?8Q�@��
C�=q@�p����
�Q����
C�J=                                    BxZ8@  
�          @�G�@S�
��z�@K�A�\)C��=@S�
�ҏ\?��HA��C�Y�                                    BxZ8#�  "          @���@5���H@r�\A�p�C��H@5�޸R?�p�AO�C��                                    BxZ82�  
�          @���@����
=@;�A��C��@����G�?8Q�@�=qC�G�                                    BxZ8A2  �          @�Q�?�ff�ȣ�@c�
A���C��?�ff���?��HA4��C�(�                                    BxZ8O�  
�          @�(�?�z�����@;�A��RC���?�z����H?!G�@���C�(�                                    BxZ8^~  �          @��H�k����R?�ff@���C�"��k������{�!�C�!H                                    BxZ8m$  
�          @�33�������H@K�A�G�C��3������\)?n{@�\C��q                                    BxZ8{�  �          @��#�
�ڏ\@~�RA�G�C�O\�#�
��
=?��HAF�RC���                                    BxZ8�p  �          A���@  �ڏ\@���A���C��)�@  ���?�p�Ab�\C�@                                     BxZ8�  "          A����߮@��\A�C�T{�������?�  AG�C��f                                    BxZ8��  
(          Aff��������@W�AîC��R������?s33@�\)C��                                    BxZ8�b  
�          A Q�?����
@���B{C�c�?����@'
=A���C�f                                    BxZ8�  
�          @�
=?8Q��˅@�ffB=qC�5�?8Q���@#33A���C���                                    BxZ8Ӯ  	�          A�?����
=@�z�A�G�C�@ ?������?���ANffC��H                                    BxZ8�T  	�          @���=�Q��Ǯ@��\B�C�e=�Q����H@{A��\C�U�                                    BxZ8��  �          @��\?W
=��  @\)A��RC��{?W
=����?�=qA]G�C�AH                                    BxZ8��  �          @�{>B�\��
=@|��A�\)C�˅>B�\��33?�p�AL��C��{                                    BxZ9F  T          @��@�z�>�(�@�33B��@���@�z�z�H@���Bz�C�B�                                    BxZ9�  
�          @�ff@�{?��@�=qB/(�A5@�{�\)@���B2�\C��                                    BxZ9+�  
�          @�@��@xQ�?�\)A��B+�@��@HQ�@<��A�Q�B�                                    BxZ9:8  �          @�@s�
@�?0��@���Biz�@s�
@�{@!G�A�\)B`�
                                    BxZ9H�  �          @�
=@;�@�(�@Z�HA��B~=q@;�@��\@���B+{Be��                                    BxZ9W�  �          @��@G
=@��?�
=An=qB�W
@G
=@�@s�
A��BuQ�                                    BxZ9f*  
Z          @�{@o\)@�{    <#�
Bs{@o\)@�(�@ ��Ar�HBnff                                    BxZ9t�  T          A{@�ff@�{�aG����
Bk33@�ff@�{?�\)AUG�Bgff                                    BxZ9�v  
�          @���@��
@ָR���_\)Bi�@��
@љ�?�ffA4��Bg(�                                    BxZ9�  
�          @��H@��@У�>��R@�\Bb=q@��@�(�@  A�p�B[��                                    BxZ9��  �          @�z�@��H@�{?�  A{B`{@��H@���@G
=A�ffBTG�                                    BxZ9�h  
�          @�Q�@�\)@��?�p�A0  B`��@�\)@��H@R�\A��BSff                                    BxZ9�  �          @�G�@�(�@���@(�A��RBU��@�(�@��@��\A�z�BA
=                                    BxZ9̴  "          A�@���@�33>��
@33BT  @���@�
=@(�A�BM�                                    BxZ9�Z  "          AQ�@�G�@�\)    <��
BZ=q@�G�@�@   A`  BU33                                    BxZ9�   �          AG�@�ff@Ϯ>B�\?��BM
=@�ff@�z�@ffAi�BF�
                                    BxZ9��  
�          A��@��@��?��\@��HB`�@��@�ff@?\)A���BV��                                    BxZ:L  
�          A�\@��R@�=q?�@c�
B;{@��R@���@�\A~�\B2��                                    BxZ:�  �          A�\@���@��?˅A/33B5��@���@�(�@P  A�z�B&ff                                    BxZ:$�  
�          A�@�Q�@��R@B�\A�Q�B=�
@�Q�@�=q@�Q�B	�B#(�                                    BxZ:3>  T          @�@�{@��\@�B�HB%  @�{@5@�  B0
=A��                                    BxZ:A�  "          @�  @�z�@�  @�z�B��B)ff@�z�@��@�33BE�HA��                                    BxZ:P�  	�          @�ff@���@�p�@��
B=qB
=@���@-p�@���B-��A�
=                                    BxZ:_0  �          A@��R@�@k�A�(�B(�@��R@J=q@�=qBA�p�                                    BxZ:m�  
�          AG�@�ff@��@Q�A�\)B*
=@�ff@tz�@��HB
z�B(�                                    BxZ:||  �          @�@�ff@333@�=qB��A�@�ff?���@�B*�A^ff                                    BxZ:�"  S          @�@�p�?G�@��\B;z�A	p�@�p��aG�@�=qB:�
C�/\                                    BxZ:��  �          @�ff@\�   @�(�B�C���@\����@���B  C��=                                    BxZ:�n  "          @�Q�@��H����@���B33C�S3@��H�:�H@��B C���                                    BxZ:�  
Z          @�{@��ÿ���@���B =qC��f@����L(�@��BQ�C�f                                    BxZ:ź  
�          @��@���@��@�z�B�C�@ @����
=@dz�A�{C��f                                    BxZ:�`  
Z          @�=q@hQ��1�@�z�BUG�C�B�@hQ���{@��\B&{C���                                    BxZ:�  T          AG�@\)��33@�Q�B#\)C�w
@\)��p�@mp�A��
C���                                    BxZ:�  
�          @�
=@�z����@l��A�=qC�~�@�z���ff@Q�A�{C���                                    BxZ; R  	�          A@�����@4z�A�p�C�@ @�����?�33A�HC���                                    BxZ;�  �          @�z�@�ff���@xQ�A��C��@�ff��G�@p�A�
=C�B�                                    BxZ;�  T          @��R@�G��Q�@�  A�
=C�=q@�G��[�@I��A�33C�7
                                    BxZ;,D  T          @��@�\)��(�@*�HA��C��=@�\)��
=?���A$(�C���                                    BxZ;:�  "          @��@����333@�  A�33C�{@����tz�@A�A�
=C�B�                                    BxZ;I�  �          @���@����;�@�p�B&Q�C�� @�������@��
A�p�C��=                                    BxZ;X6  �          @�
=@����p��@�Q�B��C���@�����z�@P��A�G�C��                                    BxZ;f�  �          @�@�����R@y��A��C�n@����_\)@B�\A�p�C��                                     BxZ;u�  �          @�p�@�Q쿠  @w
=B	G�C��)@�Q���\@VffA�(�C���                                    BxZ;�(  �          @�{@�z�>W
=@���B�H@�@�z�n{@z�HB	�RC�O\                                    BxZ;��  �          @�@�
=?��
@~{Bp�Ao�@�
=>�33@��B{@^{                                    BxZ;�t  T          @�  @Ϯ?���@vffA�Q�AH��@Ϯ>���@�33B ��@+�                                    BxZ;�  	�          @�{@�>8Q�@?\)A��R?��
@��(��@:�HA�(�C�C�                                    BxZ;��  �          A�@���@n�R?�=qAQ��A�33@���@Dz�@333A��RA��
                                    BxZ;�f  �          A=q@�G�@_\)@ffAo\)A�33@�G�@1G�@?\)A���A���                                    BxZ;�  �          A ��@��@e@Ap��A�@��@7
=@@��A���A�{                                    BxZ;�  �          @�p�@�p�@Z�H@%A�z�A��@�p�@#�
@\(�A�A�                                      BxZ;�X  �          A z�@�{@fff?�p�AG33A�p�@�{@>�R@*=qA�\)A��
                                    BxZ<�  �          @�ff@��?�ff?���AZ�HA?
=@��?z�H@�A��@�(�                                    BxZ<�  "          @��R@�  ���?�33Ak�C�C�@�  ����?(�@�(�C�/\                                    BxZ<%J  
�          @�z�@�p���{>\)?��\C���@�p���녿����G�C��                                    BxZ<3�  "          A
ff@�p�����(�����C�w
@�p������Q��eG�C���                                    BxZ<B�  
�          A(�@�  ��p��L�����C�L�@�  ��ff�"�\��G�C�b�                                    BxZ<Q<  
�          A  @�Q���p����fffC��)@�Q���33�ff�l��C�(�                                    BxZ<_�  T          A�\@������þL�Ϳ�{C��=@�����{�(��t  C��                                    BxZ<n�  "          A33@�G����;����33C�S3@�G������z��\)C���                                    BxZ<}.  
�          A
�R@E����%�����C���@E�����=q��p�C�o\                                    BxZ<��  T          A�
@J=q��{������HC���@J=q��=q�P����  C���                                    BxZ<�z  T          AQ�@�����zῇ�����C�޸@����ҏ\�>{��Q�C��
                                    BxZ<�   
�          A  @�p����
?u@���C�W
@�p����H��33��\C�b�                                    BxZ<��  T          A�
@z�H����?�@Y��C�aH@z�H���
��{�*�\C��)                                    BxZ<�l  �          A  @Z�H����?�33AIG�C��)@Z�H� (���\)��{C���                                    BxZ<�  �          A33@Z=q���H?˅A)�C��q@Z=q��\)�z��vffC��\                                    BxZ<�  �          A33@<���   ?�
=A2�\C�!H@<����\�
=q�c33C��3                                    BxZ<�^  
�          A��@Tz���p�?�  A8z�C�^�@Tz������G��:=qC�%                                    BxZ=  �          A�@������?��@h��C��3@�����ÿ����G�C�+�                                    BxZ=�  
�          A
=@����ff�u�ǮC�O\@������33�[�C��\                                    BxZ=P  T          A
{@��\���?!G�@�Q�C�ff@��\��녿����G�C���                                    BxZ=,�  	�          A	?����?�z�A1C�xR?�������\)C�]q                                    BxZ=;�  
Z          A
ff?����R?�(�AT  C�/\?���ff��\)��C�                                    BxZ=JB  T          A	p�@���z�@Q�A���C���@���
>L��?���C�k�                                    BxZ=X�  �          @�Q�@���e�0����C�޸@���Q녿�\)���\C�%                                    BxZ=g�  
�          A Q�@�  ?8Q���U�@��H@�  ?�Q�˅�6�HA33                                    BxZ=v4  �          A ��@�33?��׿:�H��ffA33@�33?\��{�=qA/\)                                    BxZ=��  T          A Q�@�R@4zᾀ  ����A���@�R@2�\?   @dz�A��
                                    BxZ=��  T          @��
@�
=@���x��A�p�@�
=@��=�?^�RA��\                                    BxZ=�&  "          @�{@�  @'���33�G33A��
@�  @=p��h�����A��                                    BxZ=��  T          @�
=@�Q�@ �׽��h��A��
@�Q�@p�?
=q@�A��\                                    BxZ=�r  �          @�  @�=q@(�?�{A*�\A���@�=q@   ?���Au�A��H                                    BxZ=�  
�          @�\@޸R?��@\)A�
=A*�H@޸R?�R@0  A��@��\                                    BxZ=ܾ  �          @���@���@@g
=A���A��R@���?��@���B��A"�H                                    BxZ=�d  T          @�ff@l(�?�\)@��
B^�A�
=@l(�=#�
@��HBk{?�R                                    BxZ=�
  
�          @��H@�
==L��@Tz�A�G�>�@�
=�O\)@N{A�33C��3                                    BxZ>�  T          @���@�녿\(�@hQ�A�33C��
@�녿�\@R�\A��
C��H                                    BxZ>V  
�          @��@������@���B?G�C��
@����[�@��B�
C��=                                    BxZ>%�  
�          @�
=@W��>�R@�BD
=C�<)@W����R@��B�C�S3                                    BxZ>4�  "          @���@Z�H��33@���B=qC��@Z�H��  @6ffAə�C�.                                    BxZ>CH  �          @�(�@#33�\)@�{B%�C�Q�@#33��@A�A�33C���                                    BxZ>Q�  T          @�@=q����@�{B�C�4{@=q�S�
@ə�Bc�C��                                    BxZ>`�  �          @��
?Q녿
=q@�B�
=C�K�?Q���@�B�aHC�Y�                                    BxZ>o:  T          @�ff?c�
�   @��HB��RC��?c�
�s33@��HBm�\C��H                                    BxZ>}�  �          @�{?�ff�1G�@�=qB��{C�k�?�ff���R@�z�BU{C���                                    BxZ>��  �          @�
=?����.{@��B�\)C�n?������@���BTffC��=                                    BxZ>�,  �          @��R@�
�#�
@��HB}
=C��@�
��ff@�\)BM=qC�k�                                    BxZ>��  �          @�ff?�Q��n�R@�G�BkC�/\?�Q����@�z�B5ffC���                                    BxZ>�x  
�          @��>��R���@���B�8RC��\>��R�i��@޸RBx��C�q�                                    BxZ>�  �          A   >��ÿ���@�  B��qC��>����n�R@ᙚBx(�C���                                    BxZ>��  "          @�{@��B�\@�=qB���C�Z�@��#33@���B{�RC��                                    BxZ>�j  T          @�z�@"�\��\@�(�B�W
C�E@"�\��@��Bz�C���                                    BxZ>�  
�          @�\)@9���^�R@��HB�(�C��\@9���)��@�z�Bq=qC���                                    BxZ?�  "          A (�@L�Ϳh��@�
=B���C�R@L���*=q@أ�Bi�HC��                                    BxZ?\  :          @�@-p��:�H@�Q�B�C�j=@-p��   @ۅBw
=C��f                                    BxZ?  
B          @�{?��H�\)@�G�B��qC�\?��H�G�@�B��qC�h�                                    BxZ?-�  �          @��@�
��@�B�B�C��)@�
�;�@ۅBu�C�                                      BxZ?<N  �          @�=q@!녿�{@�G�B�k�C�/\@!��6ff@�G�Br�RC��\                                    BxZ?J�  �          @��@i����@��
B`33C�%@i���Vff@�
=B?ffC���                                    BxZ?Y�  
Z          @��H@l(���@ə�BbffC�>�@l(��X��@���BAC���                                    BxZ?h@  �          @�=q@Dz´p�@�  B�
=C�/\@Dz��G�@�B^�C�E                                    BxZ?v�  
�          A
=@A��33@�  By(�C�ff@A��~�R@�\)BQ=qC���                                    BxZ?��  �          Aff@�ff�\(�@�G�B@�C�S3@�ff��G�@���B(�C���                                    BxZ?�2  
�          A ��@���dz�@�B,��C��{@������@��B�C��R                                    BxZ?��  T          @�{@��R�4z�@���B-�RC�1�@��R����@�=qBG�C�u�                                    BxZ?�~  T          A�@�ff�,��@��RB>�
C��@�ff����@�z�B�C���                                    BxZ?�$  �          @�(�@��H�@  @�{B.��C�R@��H���R@�=qB�C�}q                                    BxZ?��  
�          @���@�ff�g
=@��B  C��{@�ff���@j�HA��
C���                                    BxZ?�p  �          @��@��\�i��@��
B33C��@��\����@H��A�C���                                    BxZ?�  �          @��@�  ��ff@�(�B)�
C��
@�  ��R@�\)BG�C��R                                    BxZ?��  �          @���@�33�!G�@�(�B>�HC�S3@�33��\)@��\B1\)C�q�                                    BxZ@	b  �          @�ff@�ff?333@��\BHG�A�
@�ff��\@�33BIz�C��=                                    BxZ@  �          @�ff@o\)?���@��BW�\A��H@o\)���@�\)B^�C���                                    BxZ@&�  
�          @�  @�  ��\)��Q��U�C��q@�  ��p���=q��G�C���                                    BxZ@5T  �          @�(�@ҏ\���?�@��
C���@ҏ\�p�<��
>\)C���                                    BxZ@C�  !          @���@�녿�\)>���@c�
C��@�녿�=�\)?(��C��                                    BxZ@R�  �          @�z�@�ff��\?z�HAp�C���@�ff�8Q�?W
=@�
=C���                                    BxZ@aF  "          @�=q@�G�?��?�(�A�\)AI@�G�?\(�?�(�A��HA�
                                    BxZ@o�  �          @�  @��\?�33@<(�A�(�A�@��\?n{@O\)A�(�A�                                    BxZ@~�  �          @��@�(����@3�
A�  C��{@�(��fff@+�A�z�C�                                      BxZ@�8  �          @�p�@޸R?�  ?#�
@��
Aa@޸R?Ǯ?��\A�HAI                                    BxZ@��  �          @�@��
@W
=�Y����(�A��
@��
@]p��u��G�A�\)                                    BxZ@��  �          A z�@��H@vff��  �,��A��H@��H@��H�
=��
=A��H                                    BxZ@�*  �          AG�@��@\(�������Aͅ@��@j=q���o\)A�z�                                    BxZ@��  "          AG�@�(�@���
�H�z=qA�\)@�(�@3�
��=q�4��A���                                    BxZ@�v  
�          A Q�@�(�?
=��
���\@�p�@�(�?�����v=qA�\                                    BxZ@�  
�          A   @�?(����=q@�=q@�?�\)����y�Ap�                                    BxZ@��  "          A   @�\)?
=��
=�_
=@��H@�\)?�  ��  �I�@�(�                                    BxZAh  �          @�\)@�p�?���
=�u�@��@�p�?�����^�H@��\                                    BxZA  
�          @�\)@�G�>�׿�\)�;33@Z�H@�G�?O\)��(��*ff@�ff                                    BxZA�  T          A   @�{>k��k���=q?�
=@�{>�(��W
=��G�@E                                    BxZA.Z  �          @�
=@�{>�=q�#�
����?�p�@�{>�����Q�(��@�                                    BxZA=   �          A ��@�?fff@Au��@ڏ\@�>�(�@�RA��@N{                                    BxZAK�  T          A@�\)=���@=qA��R?@  @�\)��(�@Q�A��\C�j=                                    BxZAZL  �          A=q@�=q��(�@:�HA���C�c�@�=q��ff@0��A�
=C�
=                                    BxZAh�  �          A ��@��
�J=q@i��A�=qC�ٚ@��
�Ǯ@Y��A��C���                                    BxZAw�  
�          A�@�=q���@��B��C�@�=q���H@���B ��C��R                                    BxZA�>  �          A�
@�z�?(�@�B%p�@�@�z��@�{B%��C�S3                                    BxZA��  �          @�
=@��?(�@�z�B*@��@�녿\)@���B*��C�=q                                    BxZA��  �          @��H@�  ��@tz�A��C�g�@�  �p��@mp�A�33C��3                                    BxZA�0  �          A  @�(���녽��O\)C�'�@�(��|�Ϳz�H�ۅC���                                    BxZA��  �          Ap�@���k�?E�@��HC��@���p�׼��8Q�C��\                                    BxZA�|  T          A	p�@�(��^�R?�\)AH��C�@�(��r�\?��@��C��=                                    BxZA�"  �          A�R@�R�Dz�@z�A��\C���@�R�_\)?У�A2�\C�y�                                    BxZA��  �          A{@�Q��9��@�Ax��C�  @�Q��R�\?\A,��C�Ф                                    BxZA�n  �          A(�@��I��@��AzffC�%@��b�\?��RA(Q�C��q                                    BxZB
  
�          Ap�@�{�  @Z�HAÅC�L�@�{�:�H@8Q�A��HC��{                                    BxZB�  �          A��@�z��,(�@n{AۮC��
@�z��Z=q@E�A��C�j=                                    BxZB'`  T          A@�{�:=q@���B ��C�h�@�{�o\)@fffA��
C�p�                                    BxZB6  �          A Q�@��,��@���B
=C�0�@��a�@i��A��C�!H                                    BxZBD�  �          @�\)@ۅ��\)@mp�A�ffC�]q@ۅ�ff@S33A��C��                                    BxZBSR  T          A{@�z���H@c33A�G�C�T{@�z��E@>�RA�{C���                                    BxZBa�  �          A   @�G��#�
@2�\A�G�C��q@�G��E�@p�A�(�C�/\                                    BxZBp�  �          Ap�@�z��!�@�\Aj{C��@�z��8��?�p�A)G�C�Q�                                    BxZBD  �          A�@����@��Av�HC�4{@����.�R?�\)A:{C�޸                                    BxZB��  �          A�@��Q�?ǮA0��C�T{@��(��?�ff@���C�n                                    BxZB��  T          A
=@����)��?�=q@�C�u�@����3�
?�@l��C��=                                    BxZB�6  �          A33@��
�<��?�R@��\C�o\@��
�@��<��
=�G�C�9�                                    BxZB��  �          A��@�\)�>�R>L��?�{C�u�@�\)�=p������1�C���                                    BxZBȂ  �          A\)@�G��HQ��G��G
=C���@�G��>�R�����C�8R                                    BxZB�(  �          A33@�\�5���R��
C���@�\�!녿��
�H��C�                                    BxZB��  
�          A��@��+��z��g�C�=q@��{�#�
���C���                                    BxZB�t  �          A�
@��
�\(��{�{
=C��@��
�;��6ff����C��\                                    BxZC  �          A�@ڏ\�q��!G���33C���@ڏ\�Mp��Mp���ffC�e                                    BxZC�  �          A�\@�33�j=q�   ���RC���@�33�Fff�J=q����C���                                    BxZC f  T          A{@�=q��(�����{C���@�=q�e�HQ����C���                                    BxZC/  �          Ap�@���fff�`  �ܣ�C�Ǯ@���6ff��(��p�C��                                     BxZC=�  �          A=q@b�\�n{��
=�I�HC��=@b�\�����=q�f��C���                                    BxZCLX  T          A ��@z=q��������*G�C���@z=q�Q������I33C�                                    BxZCZ�  T          A (�@�ff�z=q��G���=qC��H@�ff�Dz����R��C���                                    BxZCi�  T          A ��@��R��z��g���Q�C���@��R�w
=����C�\)                                    BxZCxJ  T          A ��@����
=�~�R����C���@���xQ������p�C���                                    BxZC��  �          A ��@\)���������
C��q@\)�qG����
�;��C�W
                                    BxZC��  T          A ��@�G����\�qG�����C��
@�G���G������C��                                    BxZC�<  �          A   @������33�n�HC�xR@�������?\)���HC���                                    BxZC��  T          Ap�@ۅ���H>�
=@>�RC���@ۅ��33��33�!�C���                                    BxZC��  T          A (�@�z��]p�?��HAH(�C���@�z��n{?��@��C�Ф                                    BxZC�.  T          A ��@�\)�5?333@���C��R@�\)�:�H>.{?���C�S3                                    BxZC��  �          A   @���Mp��fff�θRC�
@���?\)��(��)�C��3                                    BxZC�z  �          A (�@�33�fff�\)���C���@�33�[����H�  C��                                    BxZC�   
�          A�@�\�.�R�aG���=qC��@�\�)���8Q���33C�`                                     BxZD
�  �          A@�R��������C��@�R��33�����HC��q                                    BxZDl  T          @�{@�(�������\)�=qC�(�@�(���{��  �\)C���                                    BxZD(  �          Ap�@��
���z�H���RC�+�@��
��(���(���RC��R                                    BxZD6�  �          A ��@�33����@H��A��RC�o\@�33�u@@��A��C�J=                                    BxZDE^  T          @���@У׾W
=�Q���ffC�3@У׽u�Y�����C��)                                    BxZDT  "          @�=q@׮?}p��n�R��p�AG�@׮?У��_\)���
AY�                                    BxZDb�  �          @��@�
=?����ff��=qA33@�
=?��H��\)�d  A6�R                                    BxZDqP  
�          @�  @��'
=���uC���@��p��p����C�%                                    BxZD�  
�          @��
@�G���\�8Q���ffC��3@�G����R�H�����C��
                                    BxZD��  �          @��H@׮�G��*=q����C��f@׮���
�>{��(�C��q                                    BxZD�B  T          @�
=@��
����33C���@��
���Ϳ�G��9�C��3                                    BxZD��  �          @�{@�{�����J�RC�|)@�{>#�
���Jff?��H                                    BxZD��  �          @�G�@��������.ffC���@��u�����2=qC���                                    BxZD�4  �          @��@��׿Y���G����C��q@��׿5�k���
=C�g�                                    BxZD��  �          @�@��׿#�
��(���HC���@��׾�(��������C�o\                                    BxZD�  �          @�ff@���@  �J=q���RC�H�@������fff��G�C��                                    BxZD�&  �          @�@�(���\���H�a�C�"�@�(���
=�����C�u�                                    BxZE�  �          @��@�ff�
=?n{@�(�C���@�ff�=p�?Q�@���C�AH                                    BxZEr  �          @�  @�  ���@*=qA��RC�aH@�  �Y��@#33A�{C��=                                    BxZE!  �          @�\)@�����녾k���\)C�@�����ff��{�
=C�\)                                    BxZE/�  �          @��@  ���H��  �z�C��f@  �����������C��H                                    BxZE>d  �          @��
?�\)��ff�ٙ��J�RC�7
?�\)��=q�:=q���
C�}q                                    BxZEM
  
�          @�\)@���
�k����C��@���
����C�l�                                    BxZE[�  T          A   @�\���
�������C�*=@�\��\)�)�����RC��
                                    BxZEjV  T          A@񙚿˅�p���33C��@񙚿��+����RC��
                                    BxZEx�  T          AG�@��H���׿�ff�Q�C��q@��H��z��G��,(�C���                                    BxZE��  �          A=q@�Q���
��\)�8(�C��f@�Q��  ����V�HC���                                    BxZE�H  
�          A	��A�c�
��{��HC���A�+���p��$��C��f                                    BxZE��  �          A	A\)>�=q�Q�����?�{A\)>��ͿE���p�@-p�                                    BxZE��  T          A
�\A��?(�þ�Q��=q@�p�A��?5��  ��
=@��                                    BxZE�:  �          AQ�A�H��33�^�R���C�#�A�H��  ��=q��{C��=                                    BxZE��  �          A=q@�
=���    <��
C��R@�
=��p��G����\C���                                    BxZE߆  �          A�R@����<��
=�C��3@���33�Tz���z�C��                                    BxZE�,  �          A�@����G��5��=qC���@�����
�\�%�C��                                    BxZE��  �          A�@�(���p��Tz�����C��@�(������  �$(�C��f                                    BxZFx  �          A	p�@����n�R���H�8Q�C���@����Z�H����t��C��                                    BxZF  �          A	@�p��qG����[
=C��f@�p��Z�H�%���(�C��H                                    BxZF(�  �          A�@�R���ÿ������C��f@�R��=q��p��<z�C�J=                                    BxZF7j  �          A33@�z��w
=��Q���C�7
@�z��h�ÿ�G��@��C��                                    BxZFF  �          A��@�G���ff�aG����
C�ٚ@�G����
�c�
���
C�R                                    BxZFT�  �          Aff@���p��=p���G�C�  @���Q쿷
=���C�w
                                    BxZFc\  T          A�R@߮���
��G��AG�C�B�@߮��Q쿏\)��{C��R                                    BxZFr  �          A(�@������H�J=q���
C��=@����{������RC�f                                    BxZF��  �          A�
@��b�\�����{C���@��Tz���H�9C�4{                                    BxZF�N  T          A
=@����  �z�H�ָRC��=@���s�
��ff�*�RC�:�                                    BxZF��  
�          A\)@�Q��I���:�H��\)C���@�Q��@�׿�
=��C�c�                                    BxZF��  �          A��@�
=��Q쾣�
��\C���@�
=��p��n{�ָRC�R                                    BxZF�@  T          Az�@g
=��p�?�(�A\��C�Y�@g
=��(�?k�@�z�C�f                                    BxZF��  T          A�@�
=����?�ATQ�C��\@�
=�߮?n{@�(�C�K�                                    BxZF،  T          A (�@�\)����>�Q�@%C�~�@�\)��Q�   �e�C��                                    BxZF�2  "          A�\@�\)��
=>�(�@@��C��=@�\)��
=��33��RC��f                                    BxZF��  �          Az�@�33��=q>�p�@&ffC��@�33��=q���R�
�HC��                                    BxZG~  �          A��@߮��z�<#�
=#�
C��@߮��33�(���
=C��                                    BxZG$  "          A33@�z��vff����VffC��
@�z��o\)��  ��33C�(�                                    BxZG!�  
�          A z�@�  �tz�5���C���@�  �l(���p����C�                                      BxZG0p  �          A ��@���}p��B�\��C�
@���tzῥ����C���                                    BxZG?  �          A�@���g���  �,Q�C�XR@���X�ÿ�p��dz�C�                                      BxZGM�  T          AG�@��
�X�ÿ����4��C�Ff@��
�H���G��h��C�
                                    BxZG\b  
�          A=q@�
=���R�Z�H����C�(�@�
=���\�e��p�C��)                                    BxZGk  �          A�@�=q�޸R�<����
=C�P�@�=q��=q�I�����C���                                    BxZGy�  T          A�@��;.{��
=��\)C�J=@���>Ǯ��ff��ff@P��                                    BxZG�T  T          A=q@�p���\)������p�C���@�p�>����
���@|(�                                    BxZG��  "          A=q@�\)>�G����
��=q@fff@�\)?z�H������{@�\)                                    BxZG��  
�          A\)@���?5�������@���@���?��\����ffA'
=                                    BxZG�F  l          A�@�?��H��33���AM�@�@33���
��A�p�                                    BxZG��  �          A�\@�(�?xQ�������A��@�(�?�  �������RAK�
                                    BxZGђ  
�          A��@�녽��
����Q�C��f@��>�ff�������R@tz�                                    BxZG�8  �          AG�@�(��u�u��\)C��@�(����{���\)C�R                                    BxZG��  �          A ��@Ϯ?h����(����@�ff@Ϯ?�����\)��RAJ{                                    BxZG��  T          Ap�@��R@   ��p��p�A��@��R@Dz���=q�(�A��                                    BxZH*  "          A z�@�@G��u����A��\@�@�R�c33�ӅA��                                    BxZH�  �          A ��@�=q?����p���
=@��
@�=q?�����=q��Q�A=q                                    BxZH)v  �          @�@�\)��=q�l(���
=C�h�@�\)��{�w
=��(�C�T{                                    BxZH8  �          @�{@��ÿ�\)�c33��C�E@��ÿ��n{��33C��                                    BxZHF�  �          @�z�@���L(����qC�xR@���:=q������
C�l�                                    BxZHUh  �          @�z�@�  �P  ������C�&f@�  �;��1G�����C�AH                                    BxZHd  T          @���@��
�@�׿�\)�=p�C��=@��
�2�\��(��g\)C�J=                                    BxZHr�  �          @�(�@���/\)�\)��ffC�Z�@������#33���C�c�                                    BxZH�Z  �          @�@�Q��\)�P  ��ffC�~�@�Q쿜(��Z�H��G�C�\                                    BxZH�   �          @��@��
�   �o\)��=qC���@��
��\��Q���33C���                                    BxZH��  T          A�@��
��=q@HQ�A�p�C���@��
���@"�\A���C��)                                    BxZH�L  �          A{@�������?��A;33C��=@�����?�{@��HC�1�                                    BxZH��  �          A�@�z�����L�;���C�q�@�z����
���p��C���                                    BxZHʘ  �          A�@�{��33>�ff@L��C��q@�{����.{���RC��3                                    BxZH�>  �          A ��@�����?ٙ�AD(�C��\@�����?�ff@�C��=                                    BxZH��  �          A z�@�p���\)@�Aj�HC���@�p���p�?��AC�H�                                    BxZH��  �          AG�@�����Q�@%�A��C��)@�����Q�?��AYG�C�J=                                    BxZI0  �          A�@{�����@��A~=qC�H�@{���33?�Q�A%p�C���                                    BxZI�  �          A ��@�ff���\@8Q�A�Q�C�k�@�ff���
@p�A�(�C��H                                    BxZI"|  �          A��@����z�@N�RA�ffC���@�����R@%�A���C��                                    BxZI1"  �          A  @�z����þ���S�
C��@�z���ff�p�����
C�N                                    BxZI?�  �          A��@�z��5�{���\C�}q@�z��#33�0����C�w
                                    BxZINn  T          A�@����)���;����
C�H@����z��L�����C�,�                                    BxZI]  �          A��@������G��  C���@����z���  �{C�                                    BxZIk�  <          A(�@ʏ\�����z��
�\C�&f@ʏ\��G�������C�=q                                    BxZIz`  
�          A��@ȣ��,(�����33C�c�@ȣ��{��=q�

=C�B�                                    BxZI�  �          A�
@߮�>�R@?\)A�C�s3@߮�Q�@*=qA�  C�o\                                    BxZI��  �          A��A Q��(�?
=q@n�RC�^�A Q���R>��R@��C�8R                                    BxZI�R  �          A��A��ff�J=q��ffC��{A���H�p����G�C��                                    BxZI��  �          A�A
=�녿������HC��A
=��ff��
=�\)C�l�                                    BxZIÞ  T          A�@��Ϳ�\)��\�d(�C�
@��Ϳ�33�
�H�r�HC�ٚ                                    BxZI�D  T          A  @�\)�j�H�ff��z�C�#�@�\)�Z=q�-p����C���                                    BxZI��  �          A�H@���B�\�6ff����C�J=@���/\)�H�����C�XR                                    BxZI�  �          A(�@�G��O\)�qG���ffC��R@�G��5��=q���C�AH                                    BxZI�6  �          Az�@�{�(���E����C��@�{�z��Tz���  C��                                    BxZJ�  �          A�
@�33�Fff�p  ��p�C�n@�33�.{�����뙚C���                                    BxZJ�  T          A��@�G��Z=q�u��ۮC�>�@�G��AG���z���\)C��q                                    BxZJ*(  �          Aff@У��Q������\)C���@У��7���ff� (�C�"�                                    BxZJ8�  T          A�R@�(��J�H�S33��z�C��H@�(��5�e��{C���                                    BxZJGt  �          A  @�=q�^�R��  ��=qC��@�=q�E������C�k�                                    BxZJV  T          A�@�����G��Q���Q�C��@������R�n{�ң�C�˅                                    BxZJd�  T          A
=@�z������ff��C���@�z��y�����H���C��                                    BxZJsf  �          A�@��
�`��������
C��@��
�B�\��ff���C��                                    BxZJ�  �          AQ�@�\)��{?B�\@���C��)@�\)��\)>#�
?�{C��                                    BxZJ��  �          A
=@�=q��33�@  ���C���@�=q��  ����  C�                                    BxZJ�X  �          Az�@��H����z��L�C�b�@��H����=q�TffC�|)                                    BxZJ��  �          A�@���~{���H�z�C��3@���b�\����C�P�                                    BxZJ��  �          A�R@�Q���z���\)�	�HC��=@�Q��l�����\��RC�                                    BxZJ�J  �          A�@��H��
=�����
=C���@��H�����G��
�
C���                                    BxZJ��  �          A(�@����xQ������(�C��)@����^{��33���C�
=                                    BxZJ�  T          A��@��������xQ���  C�.@�����G��������
C�C�                                    BxZJ�<  �          A	�@�����R�����=qC�7
@������5���33C���                                    BxZK�  T          A��@���Q��<�����C���@�����Vff����C��=                                    BxZK�  T          A	G�@������E��p�C�b�@�����H�\����=qC�<)                                    BxZK#.  �          A�@Å�~{�����C�� @Å�fff����
=C��H                                    BxZK1�  �          A
=q@�\)��33��R��p�C�xR@�\)�����(Q���G�C��                                    BxZK@z  �          A	�@ۅ��\)�
=q�pQ�C�'�@ۅ��G�� ����C���                                    BxZKO   �          A  @����=q��G���Q�C���@����
=��p���(�C���                                    BxZK]�  �          A
ff@�=q�5��p����HC���@�=q�{��(��33C��                                    BxZKll  �          A33@���*=q��
=�\)C��3@���  ��p���
C���                                    BxZK{  T          A	@��H��Q����\�{C�z�@��H��G���
=��HC�J=                                    BxZK��  �          A33@�=q>�ff���5�
@��R@�=q?n{��(��3��A(�                                    BxZK�^  �          A��@�\)�K��g
=��RC�H@�\)�8Q��vff��
=C��                                    BxZK�  T          A��@�����=q>�(�@<��C��@����ҏ\���
�
=C��q                                    BxZK��  �          A�
@��
����������HC���@��
�tz���p���(�C�H                                    BxZK�P  T          A=q@�(��fff��Q��
�C�@ @�(��N�R�����G�C��)                                    BxZK��  �          A�@�ff��\)�*�H����C�\)@�ff�����>�R���C��                                    BxZK�  �          A
=@�33��(���(��!�C��
@�33��Q���G33C�Q�                                    BxZK�B  �          A�H@��H��
=�G
=����C���@��H�~�R�Z�H����C�o\                                    BxZK��  �          A
=@ȣ���33�dz��ȸRC��)@ȣ����\�xQ��ۅC�y�                                    BxZL�  �          A\)@�z���=q�l���ѮC�o\@�z���G���  ��z�C�S3                                    BxZL4  �          AG�@�=q���
����\)C�!H@�=q��  ��\)��C�N                                    BxZL*�  �          A  @x�������z����C��q@x����\)����'z�C���                                    BxZL9�  "          Az�@�G��QG����m��C��@�G��G
=����
C�p�                                    BxZLH&  �          A�@�\)��\)�^�R��C�o\@�\)�~{�qG�����C�<)                                    BxZLV�  �          AG�@�z���  �����R�HC��3@�z��������t��C�!H                                    BxZLer  
�          A��@��R�B�\��Q��.�C�޸@��R�(����ff�633C��3                                    BxZLt  �          A�@�=q�J�H����4(�C�  @�=q�0  ��{�<
=C���                                    BxZL��  �          A33@����XQ���\)�C�>�@����AG���ff�#�C���                                    BxZL�d  T          A
=@�G������,(����C�U�@�G�����>{��  C��                                    BxZL�
  �          A�@�����z��33�O\)C��@�����  �G��xQ�C��                                     BxZL��  �          A{@�\)��p����
��(�C�E@�\)���H��33�
=C�y�                                    BxZL�V  �          A{@������=#�
>�  C��q@������;������C��                                    BxZL��  
�          A�R@��R��{?333@��\C���@��R��\)>��
@z�C���                                    BxZLڢ  
Z          A
=@�=q��@Af=qC�^�@�=q�ə�?ٙ�A:{C��                                    BxZL�H  �          A�@�  ��(�=L��>��
C�b�@�  ���
��33���C�g�                                    BxZL��  T          A	p�@�(����
>�p�@p�C��H@�(���(����L��C���                                    BxZM�  T          A�
@�p���Q�@2�\A�C��{@�p���p�@��A��HC��H                                    BxZM:  �          A  @�{���
@�A��HC��@�{��  ?�p�AYC���                                    BxZM#�  "          A(�@��\����@[�A���C��@��\��33@B�\A��RC��f                                    BxZM2�  "          A�@������@'�A���C��@����ƸR@��Ax(�C�/\                                    BxZMA,  T          A=q@{�����@���A��
C��f@{��ȣ�@z�HA�33C�                                    BxZMO�  
�          A��@\����@���A���C�q@\������@h��A�\)C��
                                    BxZM^x  
�          AG�@�Q����R@���BQ�C�'�@�Q���
=@�\)B��C���                                    BxZMm  �          Ap�@]p���
=@��
B\)C���@]p����@�G�B33C���                                    BxZM{�  T          A�@\����\)@R�\A�33C�&f@\�����@8��A�(�C��                                     BxZM�j  T          A�@U��
=@\(�A�{C��@U����@C33A��C��                                    BxZM�  T          A�H@_\)����@C�
A�z�C�5�@_\)��{@*�HA�  C���                                    BxZM��  T          A�R@Vff���@N�RA�Q�C��@Vff��{@6ffA��C���                                    BxZM�\  
�          A�H@U��  @P  A��C�@U���@7�A�C��H                                    BxZM�  
�          A�R@`�����H@&ffA���C�,�@`����
=@{Aw�C���                                    BxZMӨ  �          A�R@��
��G�@0  A���C��q@��
��@��A�ffC�]q                                   BxZM�N              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZM��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZM��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZN@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZN�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZN+�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZN:2              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZNH�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZNW~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZNf$              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZNt�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZN�p              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZN�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZN��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZN�b              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZN�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZN̮              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZN�T              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZN��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZN��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZOF              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZO�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZO$�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZO38              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZOA�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZOP�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZO_*              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZOm�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZO|v              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZO�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZO��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZO�h              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZO�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZOŴ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZO�Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZO�               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZO�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZP L              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZP�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZP�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZP,>              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZP:�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZPI�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZPX0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZPf�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZPu|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZP�"              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZP��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZP�n              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZP�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZP��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZP�`              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZP�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZP�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZP�R              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZQ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZQ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZQ%D              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZQ3�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZQB�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZQQ6              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZQ_�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZQn�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZQ}(              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZQ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZQ�t              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZQ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZQ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZQ�f              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZQ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZQ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZQ�X              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZR �              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZR�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZRJ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZR,�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZR;�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZRJ<              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZRX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZRg�  �          @��R�5��33@�Q�B��
Cnh��5����@�  B�ǮCo��                                    BxZRv.  �          @��R=�Q�c�
@��B���C��)=�Q�p��@���B���C���                                    BxZR��  �          @E��\)���
@1�B�W
Cq��\)���@1G�B�W
Cr33                                    BxZR�z  �          @���>aG��#33�b�\�Xz�C�o\>aG�� ���c�
�Z��C�w
                                    BxZR�   �          @�R?����)������{C�8R?����%�ٙ�
=C�w
                                    BxZR��  �          @�\)�W
=�HQ���33�~�C�{�W
=�Dz����
p�C��                                    BxZR�l  �          @�
=@^�R��33?�33AA��C�&f@^�R���?�\)A;�
C��                                    BxZR�  �          @��R@����׿�ff�K33C�xR@���  �˅�P  C��H                                    BxZRܸ  �          @�(�@�33�`���A����C���@�33�_\)�C33����C��)                                    BxZR�^  �          Aff@���C�
�C33��{C���@���B�\�E���p�C��)                                    BxZR�  �          A\)@�
=�vff�HQ���z�C��)@�
=�u��I����{C��                                    BxZS�  �          A�H@��R�`����=q��C���@��R�^�R���H� ��C��q                                    BxZSP  �          A��@��
�j=q�Z�H��33C��3@��
�hQ��\(��Σ�C��                                    BxZS%�  T          A=q@�=q�h���)������C�� @�=q�hQ��*�H���HC���                                    BxZS4�  �          A
=@�Q��:=q�\������C�Z�@�Q��8���]p����C�k�                                    BxZSCB  �          A�R@أ��5��]p��ə�C���@أ��3�
�^{��z�C��R                                    BxZSQ�  "          A�H@�\)�tz��J=q��Q�C���@�\)�s33�K���p�C�˅                                    BxZS`�  "          A z�@�{�W
=�fff��\)C��@�{�L���fff��\)C�/\                                    BxZSo4  T          A
=@�p������I����ffC��@�p������J=q��\)C��R                                    BxZS}�  �          A(�@�����=q�Tz�����C�b�@�������U����C�k�                                    BxZS��  �          A��@�
=�����j=q���C���@�
=��z��j�H����C��
                                    BxZS�&  �          Az�@��R�����Z�H��\)C���@��R��z��[���=qC��{                                    BxZS��  �          AQ�@�����  �:=q���C�޸@�������;����C���                                    BxZS�r  
�          A
=@������8Q����C��@������8����C��                                    BxZS�  T          A\)@Ϯ���
��R����C��@Ϯ���
�\)��p�C�f                                    BxZSվ  �          A
=@��������7�����C�}q@�����Q��8Q���G�C��H                                    BxZS�d  T          A33@�  ��=q�ff��G�C���@�  ��=q�
=���C��f                                    BxZS�
  �          A  @���������~{C�XR@����������~�RC�Z�                                    BxZT�  �          A��@�ff��G��+����HC�(�@�ff��G��+����C�*=                                    BxZTV  T          A�@˅��(����s�C�AH@˅��(����s�
C�AH                                    BxZT�  T          A��@Å��=q��G��C�C�}q@Å��=q��G��C�C�}q                                    BxZT-�  T          AQ�@���Q��G���
C��f@���Q��G���C��f                                    BxZT<H  �          A(�@�������z���Q�C��@�������z���(�C��\                                    BxZTJ�  "          A�
@�=q�ə��!���C��
@�=q����!G����C���                                    BxZTY�  �          A\)@Å�^�R>8Q�?��HC�(�@Å�^�R>8Q�?��RC�(�                                    BxZTh:  T          A  @�\����@<(�A�C��H@�\��Q�@<(�A��
C��                                    BxZTv�  T          A��@�G����@�
A�ffC�,�@�G����@z�A��\C�0�                                    BxZT��  T          Ap�@�G��s33?�=qA33C�7
@�G��s33?��A  C�8R                                    BxZT�,  �          Ap�@��
�u�?
=q@l��C�C�@��
�u�?��@p��C�E                                    BxZT��  �          Az�@ٙ���>��R@�C���@ٙ���>���@��C���                                    BxZT�x  T          A��@ᙚ���Ϳ�G��  C���@ᙚ���Ϳ�  �
�RC��H                                    BxZT�  T          A@ᙚ��p������0Q�C���@ᙚ����=q�.�HC���                                    BxZT��  �          A{@ᙚ���ÿ������C�b�@ᙚ���ÿ���  C�^�                                    BxZT�j  
�          A@�=q�U�#�
����C��@�=q�U�����ffC�R                                    BxZT�  T          Az�@��H����>B�\?�=qC�)@��H����>W
=?�(�C�)                                    BxZT��  T          AQ�@�  ��(������2�\C�g�@�  ��(��\�(Q�C�ff                                    BxZU	\  T          Az�@�z����Ϳ}p�����C���@�z����ͿxQ��׮C��\                                    BxZU  
�          A�H@�z���녿��k�C��3@�z���녿   �`��C���                                    BxZU&�  T          A{@���
=?z�@��\C���@���
=?��@���C��f                                    BxZU5N  T          A�R@��R��ff�=p���  C�Ǯ@��R��ff�5����C��                                    BxZUC�  �          A
=@����(��8Q쿞�RC���@����(�������\C���                                    BxZUR�  �          A�H@�����p�?}p�@�
=C���@������?��\@�
=C��\                                    BxZUa@  
�          AG�@�����=q@��A�(�C�,�@������@�\A�=qC�4{                                    BxZUo�  T          A��@����G�@H��A�\)C�k�@������@J�HA�p�C�w
                                    BxZU~�  T          A�@��\?��A]��C�˅@���=q?�
=Ab{C��3                                    BxZU�2  �          A ��@�����\)@X��A�{C�.@�����ff@[�A�Q�C�<)                                    BxZU��  T          @��@+�����@�Q�B��)C�� @+����\@��B�.C�G�                                    BxZU�~  T          @�  @Q��N{@�{BU�C��=@Q��K�@ƸRBV��C��)                                    BxZU�$  T          @��
@j=q�xQ�@�\)B<33C��@j=q�u@�Q�B=\)C�˅                                    BxZU��  T          @�\)@��R���
@�33B��C�  @��R���H@�(�B
=C�=q                                    BxZU�p  �          @�
=@�p�����@eA��
C�  @�p�����@hQ�A�Q�C�4{                                    BxZU�  �          A z�@�����
@ ��A�=qC��@����33@#33A���C��)                                    BxZU�  �          @�Q�@�녿�p�@�ffB;ffC��@�녿�Q�@��RB<  C�P�                                    BxZVb  T          @�
=@9��?}p�@��B���A��H@9��?�ff@�Q�B��A��H                                    BxZV  �          @�G�@��?(��@��HB�33A�@��?8Q�@�\B��A�G�                                    BxZV�  �          @��?�  >���@�\B��
Am�?�  >�@�=qB��\A��                                    BxZV.T  T          @��?�ff�333@�\B��C��?�ff�#�
@��HB���C�]q                                    BxZV<�  �          @���@G��(��@�z�Bi��C�޸@G��%�@��Bj�C�0�                                    BxZVK�  �          @�\)@�33�g
=@�\)B��C��f@�33�dz�@�Q�B
=C��3                                    BxZVZF  �          @�
=@��R��=q@^{A��C�
=@��R����@`��Aң�C�#�                                    BxZVh�  �          @�
=@�G���z�?.{@���C�޸@�G���z�?:�H@���C���                                    BxZVw�  �          A ��@�G���녽u���C��H@�G���논#�
�L��C��H                                    BxZV�8  �          A ��@�33�ȣ׿�����HC���@�33���ÿ��
����C���                                    BxZV��  T          A��@�=q����xQ���
=C�<)@�=q��z��u��㙚C�"�                                    BxZV��  T          A Q�@�{��\)��ff�+�
C���@�{��  ��  �%�C���                                    BxZV�*  �          A�@�{���?�33A>ffC�:�@�{��z�?��HAD��C�Ff                                    BxZV��  T          A��@�  ��{@{A��RC�U�@�  ��p�@!�A�  C�h�                                    BxZV�v  �          A ��@����ff?
=@�33C�  @����{?&ff@��C��                                    BxZV�  
�          A (�@��R��@'�A�z�C���@��R���@+�A�(�C���                                    BxZV��  �          @��@n{��zῪ=q�=qC�1�@n{���Ϳ�G����C�+�                                    BxZV�h  �          @�\)@����ff��H����C�@����\)����
C���                                    BxZW
  T          @�G�@aG���=q?z�H@��C���@aG����?�ff@��\C��                                    BxZW�  T          @��@\���~{@z=qBz�C�y�@\���{�@|��BffC���                                    BxZW'Z  �          @��
@���(��@�  B �C���@���%�@���B!��C��                                    BxZW6   T          @��H@�\)����@>{A�{C��@�\)���@AG�A��C�(�                                    BxZWD�  �          A ��@��\��G�@�
A��
C���@��\����@Q�A��C��                                    BxZWSL  T          A�@�
=����@
�HAy��C�\)@�
=���@�RA���C�l�                                    BxZWa�  �          A ��@�ff��G�������C�C�@�ff��G��B�\��z�C�B�                                    BxZWp�  �          @�@��\��(��8Q����C�h�@��\��z�&ff��G�C�e                                    BxZW>  "          @�(�@tz���
=�fff��RC�u�@tz������a���=qC�Z�                                    BxZW��  �          @��
?�\��=q����B�C�\?�\��z���  �?�C���                                    BxZW��  �          @��
@{�����(��B��C��q@{��\)��=q�@G�C���                                    BxZW�0  �          @�z�@\����p���G��)33C�:�@\�������\)�&�C��                                    BxZW��  T          @�{@'�������Q��RQ�C���@'����H�ƸR�O�C�K�                                    BxZW�|  �          @�33@Tz������z��*��C�>�@Tz��������\�(G�C��                                    BxZW�"  �          @��
@�ff����@����Q�C�N@�ff��ff�<(���ffC�33                                    BxZW��  �          A   @�
=��\)>k�?��C�c�@�
=��\)>�z�@�C�ff                                    BxZW�n  �          @��@У���>���@(�C��@У���p�>�Q�@'
=C��                                    BxZX  T          A (�@��H�g
=>W
=?\C��@��H�fff>��?���C���                                    BxZX�  �          A�@���l��?�ffAO\)C��@���j�H?���AT��C�                                      BxZX `  �          A Q�@�(����?�=qA8��C��@�(���z�?��A?\)C��q                                    BxZX/  �          @�(�@˅����@�\Ap��C�p�@˅���
@ffAw\)C��f                                    BxZX=�  �          @�
=@�����?޸RAI��C�xR@�����H?�AQ��C���                                    BxZXLR  �          @�
=@�(����@-p�A�
=C�q�@�(�����@1G�A�z�C��\                                    BxZXZ�  �          A   @�����  @eAծC��@����}p�@h��A�
=C�
                                    BxZXi�  �          A ��@�(��G�@�Q�Bz�C�@�(��C�
@���B	�HC�@                                     BxZXxD  �          A{@�  �9��@�ffB=qC�� @�  �5�@��B��C���                                    BxZX��  T          AG�@��
��@���B$��C��3@��
�p�@��\B%C�B�                                    BxZX��  �          A�@���@�33B:�\C��)@���@�(�B;��C�>�                                    BxZX�6  T          A�@��(�@���B��C�4{@��Q�@��B�C�w
                                    BxZX��  �          A�R@�녿�\)@��Bp�C�g�@�녿�ff@�(�B\)C��{                                    BxZX��  �          A
=@��H��@��HB6�C�q@��H���@�33B7�C��H                                    BxZX�(  T          A�@�z���@��RB�C�0�@�z��{@��B(�C�o\                                    BxZX��  T          A{@����'
=@�33B�C���@����#33@�z�B�RC��{                                    BxZX�t  T          A=q@ָR���
@�Q�A�{C���@ָR��(�@���A�p�C���                                    BxZX�  �          A
=@����z�@�ffB�
C���@�����@�
=BffC�                                      BxZY
�  �          Az�@��ÿ:�H@��B(�\C���@��ÿ(��@�  B(�C��                                    BxZYf  T          Az�@�  ���
@��B,=qC��=@�  ���H@��\B-
=C��                                    BxZY(  �          Aff@����`��@^�RA�ffC��@����]p�@aG�A�\)C�C�                                    BxZY6�  "          A=q@�
=�L(�@�z�A�\C�޸@�
=�H��@�A�G�C�3                                    BxZYEX  T          A��@ƸR����@eA��HC�~�@ƸR�~�R@h��A�Q�C���                                    BxZYS�  �          A��@���5�@aG�AǙ�C�@���1�@c�
A��C�0�                                    BxZYb�  �          Aff@��H�+�@�Q�B��C��@��H�'
=@���B33C�e                                    BxZYqJ  �          A�@�p���@���B=qC��H@�p���@��B\)C��                                    BxZY�  �          A
=@��H�{�@Z=qAθRC�
=@��H�xQ�@]p�A�=qC�33                                    BxZY��  �          AQ�@��ÿ�@�BD��C�\@��þ�@�{BD�C���                                    BxZY�<  �          A	�@��=L��@�\)Bt�
?#�
@��>��@�
=Bt@z�                                    BxZY��  �          A
{@�ff���
@�(�BA�\C�%@�ff����@���BB\)C��3                                    BxZY��  "          A	��@�{�3�
@�=qB2�HC�XR@�{�.�R@�33B4G�C���                                    BxZY�.  T          A	�@�Q��#33@�33B4�C��R@�Q��p�@�(�B5��C��{                                    BxZY��  T          A	@��H�C33@�{B0{C�&f@��H�=p�@�\)B1��C�|)                                    BxZY�z  T          A
{@���z�@��B ��C��f@����\@��
B�RC���                                    BxZY�   T          A��@�=q�\)@���B�\C��@�=q�z�H@�=qBffC��                                    BxZZ�  �          A(�@�G���@�
=B��C���@�G����@���B�\C�{                                    BxZZl            A	�@�녿���@�=qA���C���@�녿��@��HA�\C��                                    BxZZ!            A	��@��
��\)@b�\AǮC��@��
����@c�
A��C��                                    BxZZ/�  �          A\)@��Ϳ���@0��A��RC��@��Ϳ��
@1�A�C�=q                                    BxZZ>^  �          A��@�{���\@VffA��HC��@�{����@Z�HA���C�(�                                    BxZZM  �          A��@ᙚ�x��@1�A�z�C��@ᙚ�w
=@5A�p�C���                                    BxZZ[�  
�          A  @�
=�L(�@�A��C�p�@�
=�J=q@{A��C���                                    BxZZjP  �          A�@�  ���@   AZ�HC�0�@�  ����@�
Aa�C�E                                    BxZZx�  T          A�@�33���H?���AX  C��@�33���@G�A_�C��
                                    BxZZ��  T          A   @��R�$z�@�G�B��C�޸@��R� ��@��\B�HC�"�                                    BxZZ�B  q          A��@�׿�  @>{A��\C�w
@�׿��H@@  A��C��)                                    BxZZ��  7          A�@�p����@33A�p�C�f@�p���@�A�G�C�!H                                    BxZZ��  �          Az�@�
=���
@��B{C���@�
=��=q@�
=B{C���                                    BxZZ�4  �          AQ�@g�����@��A�(�C���@g���  @�A��C�R                                    BxZZ��  �          A�@�Q��Y��@�(�B��C�C�@�Q��U@�p�B  C�xR                                    BxZZ߀  �          AG�@�\)?
=q@n�RA�R@���@�\)?
=@n{A�(�@�{                                    BxZZ�&  �          A
=@�ff@*=q@|(�A�\)A�G�@�ff@-p�@z=qA�G�A�{                                    BxZZ��  T          A�@�Q�@|��@H��A��B{@�Q�@\)@EA���B�                                    BxZ[r  �          A
=@�ff@��H@C�
A�33Be��@�ff@�(�@>�RA�ffBf�                                    BxZ[  �          A{@|(�@���@���B{B\�\@|(�@��R@��\BB]��                                    BxZ[(�  T          A��@�z�@9��@�(�B��Aљ�@�z�@=p�@��HBffA��                                    BxZ[7d  T          A�\@θR���@��\B�C�XR@θR�
=q@��\B�C���                                    BxZ[F
  �          A�
@�33�&ff@�
=B:z�C���@�33��@�
=B:��C��                                    BxZ[T�  �          Az�@���    @�  BZG�=L��@���=��
@׮BZ=q?�                                      BxZ[cV  T          A�@�(��#�
@ƸRBD(�C�+�@�(����
@ƸRBD=qC��                                    BxZ[q�  �          A�H@�\)���H@�=qB-33C��3@�\)��33@��HB-�
C�!H                                    BxZ[��  "          A\)@��@{�@��\B��B�@��@~�R@�G�B��Bff                                    BxZ[�H  "          A�@�z�@=p�@�G�B&�A�@�z�@AG�@�Q�B%��A�                                    BxZ[��  "          A�\@�G��%�@��\B�HC�o\@�G��!G�@��B�HC��=                                    BxZ[��  
�          A33@�p���R@�z�Bp�C�o\@�p���@��B\)C��f                                    BxZ[�:  T          A�@�Q��4z�@��\B��C��q@�Q��1G�@��B�
C�\                                    BxZ[��  	�          AG�@�(��HQ�@�B	  C�s3@�(��E�@�
=B
�C���                                    BxZ[؆  I          A��@�Q��u@�ffB�HC�%@�Q��s33@��B=qC�O\                                    BxZ[�,   2          AG�@�(���\@�G�B0(�C�f@�(���p�@�=qB1  C�L�                                    BxZ[��   d          Az�@w���@���B��C���@w���(�@��B�C�ٚ                                    BxZ\x              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ\!�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ\0j              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ\?              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ\M�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ\\\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ\k              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ\y�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ\�N              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ\��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ\��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ\�@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ\��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ\ь              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ\�2              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ\��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ\�~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ]$              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ])p              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ]8              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ]F�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ]Ub              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ]d              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ]r�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ]�T              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ]�F              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ]ʒ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ]�8              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ^*              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ^�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ^"v              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ^1              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ^?�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ^Nh              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ^]              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ^k�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ^zZ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ^�               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ^�L              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ^Ø              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ^�>              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ^�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ^�0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ_�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ_|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ_*"  ^          A
=@�{��(�@���A�C��
@�{��z�@�Q�A�RC��\                                    BxZ_8�  	�          A
=@����(�@�\)A�\)C���@����z�@��RA�ffC�|)                                    BxZ_Gn  �          A�
@e�Ϯ@c�
A�G�C�w
@e��  @b�\A�{C�q�                                    BxZ_V  �          Az�@�������@'
=A��RC��)@������@%A�\)C��R                                    BxZ_d�  �          Ap�@qG���
=@
=A�=qC���@qG���\)@�A��RC��                                     BxZ_s`  �          A=q@�p���Q�@z�HA�p�C�^�@�p�����@x��A�  C�U�                                    BxZ_�  �          A33@�\)�~{@��BC�ff@�\)�\)@�G�B(�C�Q�                                    BxZ_��  T          A�
@�z���z�@�=qBz�C��\@�z���p�@���B��C�y�                                    BxZ_�R  "          A��@�G��:=q@�p�Bz�C�H@�G��;�@���B  C��                                    BxZ_��  �          A	G�@�G��S33@�
=Bz�C�.@�G��Tz�@�ffB�HC�
                                    BxZ_��  "          A	�@��\�_\)@��B�RC��q@��\�aG�@�z�B  C��                                     BxZ_�D  �          A
ff@����G
=@���B!�C��
@����H��@�G�B z�C��
                                    BxZ_��  �          A
ff@�(��E�@�Q�B'ffC��\@�(��O\)@�  B'33C�ff                                    BxZ_�  �          A	�@��ÿ
=@�33B+ffC�L�@��ÿ!G�@�33B+=qC�                                      BxZ_�6  "          Az�@�  ���\@���B#p�C�@ @�  ���@�Q�B#�C�{                                    BxZ`�  �          A	p�@��
���
@�Q�B!33C�T{@��
����@�  B �HC�'�                                    BxZ`�  T          A
�\@���@�z�B{C��\@���@�z�B�C��f                                    BxZ`#(  �          A�@��Ϳ(��@���A�G�C�` @��Ϳ0��@�G�A���C�<)                                    BxZ`1�  T          A{@��H��p�@S33A�  C�b�@��H��G�@R�\A�\)C�H�                                    BxZ`@t  h          A  @�@U�@)��A��RA�ff@�@S�
@+�A���A���                                    BxZ`O  �          A�@�@Fff?�  A
�RA��@�@E?��
A�A��\                                    BxZ`]�  �          A�H@�\)?�=q@@  A�z�A\z�@�\)?�@AG�A��AYG�                                    BxZ`lf  T          AQ�@���?�
=@:�HA�33Ai�@���?�33@<(�A�ffAf�\                                    BxZ`{  �          A33@��H@`�����  A�@��H@c33��z����A�ff                                    BxZ`��  T          A
=@�{@HQ���UA�33@�{@HQ��ff�FffA�p�                                    BxZ`�X  �          A��@��?fff@�A��
@�
=@��?^�R@(�A�ff@���                                    BxZ`��  f          A�@��?�R@5�A��@��@��?z�@5A�Q�@��
                                    BxZ`��  T          A Q�@��
����@l��A�\C�W
@��
��{@k�A�\)C�(�                                    BxZ`�J  �          @���@��\�?\)@�33B {C�l�@��\�B�\@��A���C�<)                                    BxZ`��  �          @��\@�  ��  @�=qA�\)C�\)@�  ����@���A��
C�/\                                    BxZ`�  �          @�=q@�p���ff@>�RA�  C��f@�p����@:=qA�\)C�l�                                    BxZ`�<  �          @��@O\)��33���x��C���@O\)���H����=qC���                                    BxZ`��  �          @�(�@����AG�@��\B�RC���@����E�@�G�B33C�w
                                    BxZa�  �          @��@�Q��vff?u@�ffC�0�@�Q��w�?h��@���C�&f                                    BxZa.  "          @�z�@���W
=�E���G�C�Z�@���Tz��HQ��ď\C��                                    BxZa*�  �          @���@����Dz�����{C���@����@����G����C�&f                                    BxZa9z  �          A
=@8�ÿ�Q���
=.C�Ф@8�ÿ�����  
=C���                                    BxZaH   �          A	�@Tz�����(��}z�C�1�@Tz��\����=qC��q                                    BxZaV�  T          A	��@W
=������w{C��@W
=�
=q��33�y�C���                                    BxZael  �          A	��@J=q�{����}
=C�n@J=q���ff�33C�7
                                    BxZat            A  @Y���4z���  �j�C�(�@Y���,����G��l��C��=                                    BxZa��  �          A�@`���8������dQ�C�E@`���0����=q�f�
C��                                     BxZa�^  �          A@u��u�ʏ\�E�RC�w
@u��n�R�����H��C��                                    BxZa�  �          A�@k��e���  �W
=C�޸@k��\����=q�Y��C�e                                    BxZa��            A
=q@��?\)��\)�
=C�0�@��5��G�.C��                                    BxZa�P  T          AQ�?�33����G��3C�޸?�33��R�{��C���                                    BxZa��  �          A(�>L�Ϳ�p��£\C���>L�Ϳ���{¥G�C�:�                                    BxZaڜ  �          AG�@o\)�<����(��U  C��R@o\)�4z���{�W��C�u�                                    BxZa�B  �          A Q�@K��z���Q��k=qC�p�@K���Q��ə��m�RC�E                                    BxZa��  �          A�H@��H����Q��A�C���@��H��z����M�C��                                    BxZb�  �          A Q�@�\)��ff@H��AîC�C�@�\)����@C33A�C��                                    BxZb4  T          A   @�(��G�@�{BW33C��{@�(��=q@�z�BTC�ٚ                                    BxZb#�  �          @���@j=q�_\)@�
=B=33C�0�@j=q�g
=@���B9��C��{                                    BxZb2�  �          @��R@�Q���
=@��
B
=C�@�Q���=q@�Q�B(�C���                                    BxZbA&  �          A   @�33��G�@XQ�AɅC��@�33���
@QG�A�z�C��)                                    BxZbO�  �          @��H@2�\��  ?@  @��C�޸@2�\���?z�@��C��R                                    BxZb^r  "          @��\@�G��{�@�G�B��C�� @�G���G�@�{B{C�]q                                    BxZbm  �          @�ff@�p���G�@~�RA��C���@�p���(�@w
=A�C�Y�                                    BxZb{�  "          @�z�@B�\���@VffA�=qC���@B�\�Ϯ@L(�A��C���                                    BxZb�d  �          @�@��H����@�Q�A���C�Z�@��H����@x��A���C��                                    BxZb�
  �          @�ff@���=q@qG�A�33C�C�@���p�@j=qA�\)C��R                                    BxZb��  �          @�z�@�  ��  @��B�C�W
@�  ���@{�A���C��                                    BxZb�V  �          @��@I������@�  BQ�C��
@I����Q�@�33B��C�U�                                    BxZb��  �          @��R@Q���p�@|(�A�C�}q@Q�����@q�A��C�Ff                                    BxZbӢ  �          @�p�@s�
�У�@Q�A{�
C�'�@s�
��=q?���Af{C��                                    BxZb�H  �          @�{@����  ���
�  C��)@���ƸR�����-�C���                                    BxZb��  �          @�p�@e�����<��
>�C�� @e����þ8Q쿪=qC��H                                    BxZb��  �          @��@z=q��  �Tz���33C��@z=q��\)���
��G�C�R                                    BxZc:  �          @�@j=q����?�G�A�\C��{@j=q��{?�ff@��C��                                    BxZc�  
�          @��
@U���\)@�RA�(�C�*=@U��ٙ�@�Ao�C�\                                    BxZc+�  T          A   @'����?��A��C�� @'���{?��@���C��{                                    BxZc:,  �          A z�@L���陚?333@�G�C���@L����=q>�@Z�HC��=                                    BxZcH�  �          A z�@���?p��@�  C���@��\?333@���C���                                    BxZcWx  �          A ��@   ��?L��@�Q�C�(�@   ��=q?\)@���C�"�                                    BxZcf  �          A (�@C33��\)?�  A,��C�q�@C33����?�G�Ap�C�aH                                    BxZct�  �          Ap�@#�
���>�33@\)C�9�@#�
��p�=���?0��C�8R                                    BxZc�j  �          A ��?}p����R�����C��=?}p���{�W
=��
=C���                                    BxZc�  �          A Q�@����Q�=�G�?L��C�ٚ@����Q������C�ٚ                                    BxZc��  �          @�\)@�����?�{A<(�C��@�����\)?���A!C���                                    BxZc�\  �          A ��@B�\��
=?�
=Ax��C�
=@B�\����?��HA\  C���                                    BxZc�  �          Ap�>�{���
��=q�G�C�>�>�{��=q��{�9�C�AH                                    BxZc̨  �          A{�'����ÿ\�-�Cz�\�'���
=����LQ�Cz��                                    BxZc�N  �          A�R�*=q��33��=q�{Cz�3�*=q��G���{�5��Cz�{                                    BxZc��  �          A��(����ff��33� z�C{��(�����Ϳ�
=� z�Cz��                                    BxZc��  �          A�H��R���׾�(��AG�C|T{��R��  �8Q���=qC|G�                                    BxZd@  �          A
=���
���>\)?}p�C��)���
����#�
���C��)                                    BxZd�  �          A�\��  ��(�?z�@��C�� ��  ����>�=q?�
=C���                                    BxZd$�  �          Aff?�  ��Q�k����C�Y�?�  ��  ���tz�C�\)                                    BxZd32  �          A?�{��p�=#�
>��C���?�{��p���\)���HC���                                    BxZdA�  �          A�?�z���ff�L�Ϳ��C��?�z������j=qC��                                    BxZdP~  �          A�H?�z������\�fffC��3?�z����\�Tz����
C��R                                    BxZd_$  �          A�\?^�R� Q�B�\����C��?^�R� (����j�HC�)                                    BxZdm�  �          A33@ ����?}p�@�\)C��@ �����R?+�@�ffC�H                                    BxZd|p  �          A��@\)��p�?��
A��C�  @\)��
=?u@׮C���                                    BxZd�  �          A�?Y����{?�AL��C�\?Y��� (�?�(�A%C��                                    BxZd��  �          A���\)���?�  @ᙚC����\)�=q?&ff@�=qC��                                    BxZd�b  �          A\)>�=q��{��33�W\)C��
>�=q��33�  ��C���                                    BxZd�  �          A33?ٙ����
���R�  C��?ٙ���녿˅�3�
C�&f                                    BxZdŮ  �          A�@{���ÿ�p��`��C�{@{��{�z���  C�0�                                    BxZd�T  �          AG�?�{��
=���
���C�Ǯ?�{��������33C��{                                    BxZd��  �          A ��@\(���{�
�H�{
=C�/\@\(��ڏ\�\)��ffC�]q                                    BxZd�  �          Ap�@�  �׮��33�Y��C�W
@�  �����p��~=qC��                                    BxZe F  �          A��@�ff��  ��(��(  C��3@�ff��p�����L��C��                                    BxZe�  �          A�R@u���\���� ��C�5�@u���׿�(��'
=C�P�                                    BxZe�  �          A{@��
��ff�
=q�uC�T{@��
��p��aG���
=C�c�                                    BxZe,8  �          AG�@g
=���
�n{��z�C�t{@g
=��녿��
�{C���                                    BxZe:�  �          A=q@�  ��녾��W�C��@�  ���ÿL�����C�H                                    BxZeI�  �          A�\@�=q�������ffC��3@�=q��G��'
=���C��                                    BxZeX*  �          A�@������{���C��)@�����333����C�                                      BxZef�  �          @�\)@�Q��˅��(��{C���@�Q���G�����4Q�C��3                                    BxZeuv  �          @��@[���(���p��J�\C�B�@[��������t��C�l�                                    BxZe�  �          A ��@&ff�(��?��
A��C��q@&ff�=p�?��RA��HC�                                    BxZe��  �          A��@���@AG�@��HBJ
=B(�@���@+�@ϮBPB �\                                    BxZe�h  T          A�@��
@
=@θRBOG�A�  @��
?�G�@��BT  A���                                    BxZe�  �          A=q@�=q?��@�{BWffA_\)@�=q?J=q@�  BY�AQ�                                    BxZe��  T          A�\@�p�>W
=@�  BI�@�
@�p���@�  BI��C�C�                                    BxZe�Z  �          A=q@��!G�@��HBE�\C���@��xQ�@�G�BC��C���                                    BxZe�   �          A�R@�z�?޸R@x��A�\)Ac\)@�z�?��
@~�RA�G�AG�                                    BxZe�  �          A=q@��ÿ�(�@�B��C�Q�@��ÿ��R@��\B(�C�9�                                    BxZe�L  �          AG�@���h��@��B��C�z�@����Q�@���BG�C�&f                                    BxZf�  �          @���@�ff��p�?�=qA�HC��@�ff���?�ff@�z�C���                                    BxZf�  �          A   @�33���@�Q�B\)C��
@�33��
@���B\)C���                                    BxZf%>  �          AG�@Å�O\)@�{B �C�4{@Å��\)@�(�BffC���                                    BxZf3�  �          A�R@�����
@��RB){C���@������@��
B%�
C�\)                                    BxZfB�  �          Aff@�z��Z=q@`  A�Q�C��@�z��g
=@R�\A��C�=q                                    BxZfQ0  �          Aff@љ��qG�@�A�G�C��@љ��z=q@��Az�\C���                                    BxZf_�  �          AG�@��R�
�H@��B  C�f@��R�(�@�
=B
��C��q                                    BxZfn|  T          A=q@��R�aG�@���B��C�1�@��R�r�\@���B \)C�:�                                    BxZf}"  �          Aff@���G�@�=qBffC�R@���XQ�@��A�
=C��                                    BxZf��  �          A=q@��
��p�@5A�z�C�e@��
���H@!G�A�\)C��=                                    BxZf�n  �          @�(�@��
���@S33A�
=C��f@��
���@<��A�C�'�                                    BxZf�  �          @�ff@�����@s�
A�Q�C���@������@`  A�  C��\                                    BxZf��  �          @���@�33�<��@��RBp�C��@�33�P  @�Q�B  C�\)                                    BxZf�`  �          @�@���c�
@���B/��C�XR@����p�@��RB,C���                                    BxZf�  �          @�ff@��
�.{@�
=B
��C���@��
�@  @���B  C�o\                                    BxZf�  �          A (�@�{�P  @,(�A��RC�\@�{�Z�H@p�A�33C�xR                                    BxZf�R  �          @�\)@�G�����@��A�G�C��3@�G���p�?���AeG�C�&f                                    BxZg �  �          A ��@�p��a�?Q�@��\C��@�p��e�?�@�33C���                                    BxZg�  �          @�{@�G���G�?^�R@ʏ\C�}q@�G����H>��H@c�
C�^�                                    BxZgD  �          A�@Å��  >���@9��C�\)@Å��Q�=L��>���C�Q�                                    BxZg,�  �          A�\@�z�����@z�Aj�RC�� @�z�����?�\AH  C�8R                                    BxZg;�  �          A@�33�e�@{A�z�C�5�@�33�n{?�(�Ab�RC��)                                    BxZgJ6  �          A (�@�
=��
=@`  Aљ�C�7
@�
=��@X��Aʏ\C�B�                                    BxZgX�  �          A�@�
=@#33@��
B��A��
@�
=@(�@�G�B&�\A�{                                    BxZgg�  �          A�\@��R@��@��RBG�BCz�@��R@���@�=qB�B;��                                    BxZgv(  �          A
=@��?c�
@aG�A�z�@޸R@��?!G�@dz�A��@�
=                                    BxZg��  �          A�@�ff>L��@z=qA�=q?Ǯ@�ff��Q�@z�HA�z�C���                                    BxZg�t  �          A�\@�ff?�33@��A��
A]�@�ff?�=q@�p�B��A3�
                                    BxZg�  �          A��@��H?   @�
=B!Q�@��@��H>�@�  B"=q?��\                                    BxZg��  �          A z�@�����@�=qB3��C��f@����fff@�Q�B1�
C�h�                                    BxZg�f  �          @��@����z�@�G�BR
=C���@�����@��BL=qC���                                    BxZg�  �          A�@�ff���R@�ffB,��C���@�ff���@���B&G�C�                                      BxZgܲ  �          A Q�@��׿�\@θRBY�C�� @������@ə�BR  C�                                    BxZg�X  �          AG�@s�
����@�G�Bi=qC�q�@s�
����@�p�Bbz�C�7
                                    BxZg��  �          A�@�ff>aG�@�33Bi�H@AG�@�ff���
@�33Bi�C�ٚ                                    BxZh�  �          @�{@e���  �=p�����C���@e���p������RC���                                    BxZhJ  �          @�
=@o\)��G�������
C�q�@o\)�������9��C��H                                    BxZh%�  �          @�ff@*=q��{@(Q�A�{C�}q@*=q��(�@z�As\)C�<)                                    BxZh4�  �          @��?�=q���H@A�A�G�C���?�=q���@{A�p�C�H�                                    BxZhC<  �          @��@��\�θR�aG��ʏ\C��=@��\���
��33�"{C��q                                    BxZhQ�  �          @��@������þB�\���C���@����Ϯ�:�H��G�C�3                                    BxZh`�  �          @���@h����
=?�ffA�RC�33@h�����?=p�@��C��                                    BxZho.  �          @��\@^{�ڏ\?ǮA8��C�w
@^{��?}p�@��C�H�                                    BxZh}�  �          A z�@g
=��녿
=q�x��C���@g
=�߮������HC��=                                    BxZh�z  �          Ap�@�z���33�s33��G�C���@�z���  ���
�/
=C��                                    BxZh�   �          A�@\(���Q���H�aG�C�\@\(�����#33��C�aH                                    BxZh��  �          Ap�@j=q���*�H���HC�\)@j=q��p��O\)��G�C���                                    BxZh�l  �          A (�@^{��p��33��{C��H@^{��ff�7���G�C�(�                                    BxZh�  �          @��?�����{��R��  C�5�?�����
=�7����C�W
                                    BxZhո  �          @���@���=q@O\)A�G�C�ٚ@����H@6ffA���C��)                                    BxZh�^  �          @�G�@�33��\@�  B@p�C���@�33�G�@�=qB8��C�q�                                    BxZh�  �          @�p�@����G�@�\)B[��C��@���}p�@�p�BXQ�C���                                    BxZi�  �          @�ff@x��=�G�@��
Bn(�?�{@x�þ��H@�33Bm(�C�p�                                    BxZiP  �          @��
@x��>�@�{Bj�\@��H@x�ý���@�ffBk��C�B�                                    BxZi�  �          @�ff@AG���������z�C��3@AG������(����\)C�H�                                    BxZi-�  �          @��@�\�x������C���@��{���  C�,�                                    BxZi<B  �          A{@ ����Q���z�����C���@ ����33��\)��C�/\                                    BxZiJ�  �          Ap�@�R�ə���=q�G�C���@�R���
��z��{C�o\                                    BxZiY�  T          @�?����{��
=�
�C��\?����������   C��H                                    BxZih4  "          @�G�@����C�
>B�\@
�HC�h�@����C�
���Ϳ�{C�c�                                    BxZiv�  �          @�{@��
@XQ�@{�B��B�
@��
@?\)@�\)B(�A�\                                    BxZi��  �          @�\)@�{����Q�L��C��f@�{���Ϳ�����C��q                                    BxZi�&  �          @�G�@��
��Q����C�
C���@��
��{�}p���33C���                                    BxZi��  �          @��\@�(����R�����p�C��f@�(����\���F=qC���                                    BxZi�r  �          @�\)@��
���@0��A�  C���@��
����@  A�  C���                                    BxZi�  �          AG�@z���Q��\�F
=C���@z���z��У��Z��C�B�                                    BxZiξ  �          A z�@Dz����R��ff��RC�!H@Dz���
=����#�\C�=q                                    BxZi�d  �          @��
@����R�\?�{Ae�C���@����]p�?��
A:�HC��                                    BxZi�
  �          @�@�\)�z�@#�
A���C�n@�\)�z�@�A�G�C�z�                                    BxZi��  �          @�\)@�z�����>L��?��RC�{@�z����þu��  C��                                    BxZj	V  �          @�
=@�(����R��ff�  C��@�(���녿�  �L��C���                                    BxZj�  �          A ��@�
=��p���ff�O
=C��{@�
=���R�\)���C�xR                                    BxZj&�  �          @�
=@ڏ\�@��?�G�A7\)C��@ڏ\�I��?�Q�Az�C��H                                    BxZj5H  �          @��@���>{@ffA�33C���@���L��@�Ak�
C��                                     BxZjC�  �          @�\)@�(���@QG�A�(�C���@�(��'
=@@��A�{C���                                    BxZjR�  �          @��R@�33�O\)@A�33C�Y�@�33�]p�?�p�Ag�
C��{                                    BxZja:  �          @���@��XQ�@\)A��C���@��fff?�\)A]�C�ٚ                                    BxZjo�  �          @���@��
�c�
@�A�33C�޸@��
�s33@�An�RC�{                                    BxZj~�  �          @��
@�
=���׿@  ���RC��H@�
=�������#\)C��                                    BxZj�,  �          @�{@��
��(��fff�ӅC���@��
��  ���
�4Q�C��
                                    BxZj��  �          @��R@�G����������{C��@�G���  ����PQ�C���                                    BxZj�x  �          A ��@�=q�1G�?z�H@�G�C��R@�=q�7
=?+�@���C��f                                    BxZj�  �          A�\@�33���H?�{A�C��q@�33����?�z�A�C�aH                                    BxZj��  �          A�R@��;�Q�?�G�AIG�C��3@��Ϳ��?��HAC\)C�H                                    BxZj�j  �          A�H@��Ϳ!G�?�z�AY�C��q@��ͿTz�?�=qAO�
C��)                                    BxZj�  �          A(�A �׿�ff?�z�A�C�ffA �׿�?�G�@�G�C���                                    BxZj�  �          A�@�ff�����p��C��@�ff�33��p��$z�C��f                                    BxZk\  �          A
=@��
�"�\��p��'33C���@��
�ff��G��G
=C�h�                                    BxZk  �          Aff@ʏ\���R@1G�A�  C�/\@ʏ\��  @G�A�(�C�K�                                    BxZk�  �          A  @ۅ�q�@�A�ffC��{@ۅ����?��ATz�C��\                                    BxZk.N  �          A�@�ff��33?���@�\)C�� @�ff��{?�@l(�C�|)                                    BxZk<�  �          A(�@�z���\)?��HA\(�C��@�z���p�?�
=A\)C�n                                    BxZkK�  �          A(�@�����p�?��AG�C�p�@�������?&ff@�=qC��                                    BxZkZ@  �          A(�@��p  ?^�R@�33C�L�@��u�>�
=@=p�C��                                    BxZkh�  �          A(�@���x��?��
@�p�C���@���\)?\)@x��C�o\                                    BxZkw�  �          AQ�@�=q���
?�z�A8��C��@�=q��G�?�\)@��C�0�                                    BxZk�2  �          A��@��H���
?
=@�=qC���@��H���<�>L��C��                                     BxZk��  �          A��@�G���(�?aG�@��
C��R@�G����R>��
@��C���                                    BxZk�~  �          A��@�
=�\)@z�Ag�C�q@�
=���R?���A/33C�o\                                    BxZk�$  �          A=q@����G�@��A�33C��f@������?�AIG�C��                                    BxZk��  �          A@��H��\)@�HA��C�� @��H��  ?���AMp�C�)                                    BxZk�p  �          A��@ҏ\��G�?�AH(�C���@ҏ\��\)?���Az�C�p�                                    BxZk�  �          A��@��
����@��A�  C���@��
����?�33AT��C�                                    BxZk�  �          A=q@��H��녿.{���RC�9�@��H��{������C���                                    BxZk�b  �          Aff@�ff��=q��G��
�RC�#�@�ff���
�����M�C��3                                    BxZl
  �          A�\@�=q���ÿ��R�Q�C�J=@�=q���\��{�M�C��
                                    BxZl�  T          A33@�Q���\)���
�(  C���@�Q���  ���iC�.                                    BxZl'T  �          A
=@��������:=q����C�  @������
�a���33C�Q�                                    BxZl5�  �          A�H@����p��&ff��{C�4{@�������H����C�W
                                    BxZlD�  �          A=q@�  ��
=�O\)��{C��@�  ��Q��tz��ۮC�O\                                    BxZlSF  �          A��@�p���=q�R�\���RC��@�p����H��Q���C�P�                                    BxZla�  �          AG�@�z����H�.{���HC��@�z����a�����C�                                    BxZlp�  �          AG�@\)���������p�C��3@\)��������p�C�0�                                    BxZl8  B          AG�@�\)��{��G��QC��H@�\)���������
C�p�                                    BxZl��  �          A�
@�G������Y����  C���@�G�������=q���C��)                                    BxZl��  �          A
=@������R�����p�C�@�����33���Q�C��3                                    BxZl�*  �          Ap�@�\)��Q쿂�\�陚C���@�\)��=q��
=�A�C�q                                    BxZl��  �          A�@�z���ff�����C�Ф@�z���\)��(��b=qC�y�                                    BxZl�v  �          A@�
=���H�=p���
=C���@�
=��z��hQ���z�C��R                                    BxZl�  �          Ap�@��R��
=�R�\��  C��@��R��
=��  ��\)C�|)                                    BxZl��  �          A=q@�  ��G��`  ��33C�Q�@�  ��Q���\)���C�                                    BxZl�h  �          A=q@��
��
=�qG���33C���@��
��z������=qC�,�                                    BxZm  �          A��@������R�{�����C�<)@����������33C��
                                    BxZm�  �          A�@�����
=�����33C���@�����33��  �  C�\)                                    BxZm Z  �          A Q�@��H��
=�
=�s�
C�1�@��H���
�=p���33C�H                                    BxZm/   �          A (�@�
=��p����Q��C���@�
=��33�*=q���C�l�                                    BxZm=�  �          Ap�@��H��=q�7
=��=qC�33@��H����g���G�C�o\                                    BxZmLL  �          A��@�z���=q�^�R��Q�C�~�@�z������������C��                                    BxZmZ�  �          A@�����{�^{�ʏ\C��@�����z���������C�<)                                    BxZmi�  �          A�@q������p  �ޏ\C��@q������\�
C�q�                                    BxZmx>  �          A   @����\)��33�^�RC��R@�������,����ffC��3                                    BxZm��  �          @�\)@��R���׾u��\C��)@��R��p��������C�:�                                    BxZm��  �          @��
@���w�@�  B�C�Ǯ@����{@h��A�p�C��R                                    BxZm�0  T          @��R@����O\)@��B�C�Y�@����u@uA��C�.                                    BxZm��  �          A Q�@�Q��C33@[�A�=qC�t{@�Q��aG�@<��A�C���                                    BxZm�|  �          A   @�p��G
=@���BQ�C�"�@�p��mp�@qG�A�RC��R                                    BxZm�"  �          @�
=@�ff�`  @��B z�C�Z�@�ff��
=@���Bz�C��                                    BxZm��  �          @�ff@�z���p�@��B�C�e@�z���33@�{A�\)C�Y�                                    BxZm�n  �          @�@�{��Q�@��RBG�C���@�{���R@��B��C�o\                                    BxZm�  �          @�
=@�ff��G�@~�RA�(�C���@�ff���\@Mp�A���C�n                                    BxZn
�  �          @�\)@�=q����@�Q�A��C���@�=q���R@P��A�p�C�                                      BxZn`  �          @�
=@�  ���@s�
A��C�t{@�  ��(�@?\)A��RC�{                                    BxZn(  �          A ��@������
@��A��
C��f@�����{@W
=A��HC��                                    BxZn6�  �          A�@��\���?�\)A:{C���@��\��  ?@  @�33C�{                                    BxZnER  �          A ��@�p���>��?�ffC�8R@�p���(��E���\)C�U�                                    BxZnS�  �          A   @�p���33���W
=C��=@�p����׿z�H��\C��                                    BxZnb�  �          @�{@�33���?�p�A,z�C��@�33����?��@\)C�%                                    BxZnqD  �          @�z�@����ÿ���G�C��q@�������
�u�C�(�                                    BxZn�  �          @�z�@����z�k���ffC���@����p���z��`��C�                                    BxZn��  �          @�(�@�{��33��R��33C�}q@�{��p���\)�@  C��                                     BxZn�6  �          @�
=@`�����
����� =qC�H@`����z������C��R                                    BxZn��  �          @�33@ƸR��z�?z�@�  C��)@ƸR��p��.{��  C���                                    BxZn��  �          @��H@ʏ\���ý�G��Q�C�4{@ʏ\���R�Tz�����C�o\                                    BxZn�(  �          @�p�@������R�.{��ffC��{@�����G������G�C�W
                                    BxZn��  �          @�
=@�{�@  ��{�ffC���@�{�/\)��=q�Up�C��{                                    BxZn�t  �          @�\)@����e��(���33C�ٚ@����H���?\)���HC�]q                                    BxZn�  �          @��R@��H�U�333��{C���@��H�5�S�
���C�T{                                    BxZo�  �          @�p�@�  �p  ��=q�8��C�z�@�  �\(��
�H��C�~�                                    BxZof  T          @��@�=q�-p��{���HC��@�=q�G��8�����C�q                                    BxZo!  �          A z�@�녿���0�����C�!H@�녿���AG�����C���                                    BxZo/�  "          A{@�Q�>��
�>{��ff@�H@�Q�?L���8Q����R@��
                                    BxZo>X  �          Ap�@��
?5�B�\����@�Q�@��
?��H�8Q����RA��                                    BxZoL�  �          Ap�@�{?8Q��A����R@���@�{?��H�7�����Az�                                    BxZo[�  �          A ��@�\>��(������@fff@�\?h���!����\@��
                                    BxZojJ  �          A@��Ϳ�=q����Q�C�  @��Ϳ(���!G����C���                                    BxZox�  �          A�@�
=�\���R�c
=C�l�@�
=��z��{�}C��
                                    BxZo��  �          A@�\)�   �������C��H@�\)�޸R���=C���                                    BxZo�<  �          AG�@�{��Ϳ������HC�@�{��p����H�&�RC�Ǯ                                    BxZo��  �          A ��@�����Q����C�|)@����
�H���H�
ffC�
                                    BxZo��  �          A��@�(��=q��Q���C�9�@�(��
�H�˅�5p�C�\                                    BxZo�.  �          A�H@�33���:�H���
C�o\@�33��(���=q��(�C���                                    BxZo��  �          A33A Q쿹���W
=���C�� A Q쿣�
������G�C�t{                                    BxZo�z  �          A�\@�p���ff��33�{C��
@�p����H�(������C��                                    BxZo�   �          AG�@�z����?}p�@��C�N@�z��!�?\)@~{C��
                                    BxZo��  �          @�
=@�ff��?8Q�@��RC���@�ff��\>��@@��C���                                    BxZpl  �          A�HA녿k�>L��?�\)C��fA녿p��<�>aG�C��{                                    BxZp  �          A�\Aff��Q�>�z�@33C���Aff����>aG�?�=qC���                                    BxZp(�  �          A�\Ap����
?G�@�  C��HAp���ff?5@���C�h�                                    BxZp7^  �          A�@�Q�@%�?fff@�(�A���@�Q�@Q�?��A\)A�G�                                    BxZpF  �          A ��@���@Mp�?c�
@���A�=q@���@@  ?���A'33A�
=                                    BxZpT�  �          @��@�
=@��?��AA�  @�
=@�?ٙ�AEA}p�                                    BxZpcP  �          @�z�@�p�?��\?��
A{@��@�p�?E�?�Q�A)p�@��                                    BxZpq�  �          @��@�{?��?���A(�@�z�@�{?W
=?�ffA��@�Q�                                    BxZp��  �          @�z�@�?��?\A1�A��@�?��\?�p�AK\)@�33                                    BxZp�B  �          @��
@�ff?�@33Aq�A,��@�ff?��\@�A���@��                                    BxZp��  �          @���@��
?p��@�\A�\)@�@��
?   @�A���@xQ�                                    BxZp��  �          @���@陚?
=@#33A��@��
@陚=�G�@'�A�?\(�                                    BxZp�4  �          @���@���>B�\@>{A���?�  @��;�p�@=p�A��C���                                    BxZp��  �          @�Q�@��?\)@
�HA�=q@��@��>��@\)A�ff?�
=                                    BxZp؀  
�          @�\)@ᙚ?�33@5�A���A3�@ᙚ?Y��@B�\A��@ۅ                                    BxZp�&  "          @��R@�Q�?�  @@  A��A�@�Q�>�
=@H��A���@\(�                                    BxZp��  �          @�ff@��H?(�@:=qA�(�@�{@��H=u@>�RA�=q>�ff                                    BxZqr  
Z          @�p�@�p�>�
=@*=qA�ff@W
=@�p���Q�@,(�A�ffC���                                    BxZq  �          @�p�@�G�����@�
A��RC���@�G���@  A���C��=                                    BxZq!�  �          @��
@�
==�\)@C33A�
=?
=@�
=��\@@��A�Q�C��=                                    BxZq0d  	�          @�R@�33�8Q�@3�
A�\)C��q@�33���R@'�A��HC���                                    BxZq?
  
(          @��
@����33?�p�A=�C���@����33?��A
=C��H                                    BxZqM�  T          @�G�@�(��\��>��@�C���@�(��\(������K�C��R                                    BxZq\V  T          @�=q@�
=�s�
�   �x��C�B�@�
=�h�ÿ�(����C��3                                    BxZqj�  �          @�33@�=q�@�׿s33��  C��\@�=q�1G��\�@  C��\                                    BxZqy�  
�          @�\@�=q���\�Tz���G�C��@�=q���H��z��Q��C�xR                                    BxZq�H  �          @��
@�G���z��G���{C�Z�@�G�����8Q�����C��f                                    BxZq��  �          @�G�@�ff��Q����h(�C��@�ff�u��%���
C�y�                                    BxZq��  �          @�  @�p��~�R�#�
��(�C�\@�p��qG���z��5G�C��                                     BxZq�:  "          @陚@�=q��G�=�\)?�C�h�@�=q��
=�Q���=qC���                                    BxZq��  �          @�Q�@�\)���H@G�A��C��)@�\)���?�
=A  C���                                    BxZqц  �          @��@ȣ��&ff@Q�A���C�� @ȣ��>{?�=qAL��C�U�                                    BxZq�,  "          @�=q@�(��7
=@ ��A�Q�C��=@�(��Mp�?�33A1��C���                                    BxZq��  �          @�@�p��
=?�@�ffC���@�p���H=�Q�?.{C�W
                                    BxZq�x  T          @�{@�\)��  �@  ����C�!H@�\)���ÿ��
���C�Ф                                    BxZr  S          @�{@�\��
=��(��6�HC�Q�@�\���Ϳ��
�]C��R                                    BxZr�  T          @�@�녿�=q����>�\C��q@�녿�p���\)�i�C�{                                    BxZr)j  
�          @�33@��
��  �\�8(�C�"�@��
�+���Q��MG�C�c�                                    BxZr8  �          @��@�p����Ϳ\�6�HC���@�p��E����H�N=qC�                                    BxZrF�  T          @�p�@�ff�
=��
=�J=qC���@�ff�u��G��T��C��                                    BxZrU\  "          @�33@�
=��p���\����C���@�
==�G��z�����?^�R                                    BxZrd  �          @�\@�\��\)����$(�C���@�\�Q녿���=�C��\                                    BxZrr�  
�          @�
=@���p��}p����
C�P�@�녿��H���/�
C�@                                     BxZr�N  
�          @���@�z��*�H�:�H����C�l�@�z��p����\���C�0�                                    BxZr��  �          @���@�33�1녿����%G�C���@�33��Ϳ�33�i��C�.                                    BxZr��  
�          @��@�=q�'���
=�N�HC�� @�=q�p������C��                                    BxZr�@  T          @�  @Ϯ�U���33�Lz�C�e@Ϯ�:�H��
��(�C��                                    BxZr��  �          @�R@ᙚ�	���:�H���
C���@ᙚ������z��  C�Ff                                    BxZrʌ  
�          @���@�=q�(������-�C�#�@�=q�ff��\)�j�\C�q�                                    BxZr�2  T          @��
@����A녿Tz���\)C��q@����2�\�����5�C���                                    BxZr��  �          @�(�@ȣ��s�
�����#33C�\)@ȣ��i����\)���C��                                    BxZr�~  "          @���@�Q��w����R���C�(�@�Q��mp��������C���                                    BxZs$  T          @���@�G��r�\�#�
��ffC�y�@�G��dzῷ
=�2�\C�9�                                    BxZs�  "          @��@�{�a녿\(���C��f@�{�P�׿˅�F�HC���                                    BxZs"p              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZs1              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZs?�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZsNb              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZs]  v          @��@���������R�}�C��\@������
�@���\C�~�                                   BxZsk�  �          @�\@��H��녿�\)�l��C�*=@��H�s�
�0������C�޸                                   BxZszT  T          @�@�����\����mp�C�z�@���dz��.{��(�C�:�                                   BxZs��  �          @��
@�  �n�R����z�C��@�  �L���333���C���                                   BxZs��  �          @�\@���(���{��{C�˅@����0  ��G�C��
                                    BxZs�F  "          @���@���|(��=q��  C�W
@���S�
�N{�ә�C���                                    BxZs��  
�          @���@ƸR�7
=�p���Q�C��q@ƸR�  �A���33C��                                    BxZsÒ  "          @��@�녿��Ϳ�{�nffC�c�@�녿�33�
�H���C�7
                                    BxZs�8  
�          @�G�@�\)=L�;Ǯ�E>�(�@�\)>\)��p��:�H?�{                                    BxZs��  �          @�G�@�Q�B�\�B�\��G�C�>�@�Q�\)�k���ffC�n                                    BxZs�  T          @�@�ff��\���
�!G�C���@�ff��׾B�\��G�C�q                                    BxZs�*  
�          @��@ٙ����J=q���C��=@ٙ��˅��
=���C�k�                                    BxZt�             @�ff@�{�9����z��4��C�� @�{� �������HC�Z�                                    BxZtv  
�          @�{@ƸR�P  ��  �A�C�5�@ƸR�4z��(���ffC��f                                    BxZt*  
�          @�\)@����9����z��TQ�C��\@�������G�����C��                                    BxZt8�  
�          @���@����|������z�C�:�@����W��=p���ffC�T{                                    BxZtGh  T          @��@�(������������C�@�(��hQ��W
=��G�C�]q                                    BxZtV  
J          @�ff@��H��{�5��Q�C�J=@��H�[��n{���C�                                    BxZtd�  d          @�Q�@�p��c�
�z=q�G�C�\@�p��$z����
�33C�5�                                    BxZtsZ  
�          @���@�33�u��^�R��33C��)@�33�;�������C�p�                                    BxZt�   T          @�@�z����
�G�����C�Ф@�z���
=�Tz���p�C�޸                                    BxZt��  
�          @��
@�33�����	����(�C�{@�33�����J�H��\)C��                                    BxZt�L  
Z          @��
@�����
=q��z�C�b�@�������O\)�ڣ�C�XR                                    BxZt��  
�          @�33@�z���G��У��V=qC�E@�z���G��(�����C��=                                    BxZt��  
�          @��H@�\)�k���G��h��C��@�\)�J�H�$z�����C��f                                    BxZt�>  "          @�=q@�Q�����Q���C�y�@�Q��l(��U���p�C���                                    BxZt��  �          @�
=@Vff�����Tz���  C��@Vff��(�������C��=                                    BxZt�  
Z          @߮@S�
���H�A���(�C��=@S�
��\)��ff�
=C�>�                                    BxZt�0  T          @��@H����33�1���
=C���@H����G������{C��q                                    BxZu�  "          @�Q�@6ff��G��L����\)C��H@6ff��(�����=qC��3                                    BxZu|  �          @�Q�@p���Q��c33��C��@p�������  �*�HC�C�                                    BxZu#"  �          @�Q�@p���(��o\)�{C�Ff@p����H�����2  C��                                     BxZu1�  �          @޸R@�{��>�@�z�C�
=@�{��p�<#�
=�Q�C���                                    BxZu@n  �          @��@�\)?�@J=qA�ff@�ff@�\)�L��@Mp�AٮC�3                                    BxZuO  
�          @��@ҏ\���>��@s33C��q@ҏ\�(����z�HC��{                                    BxZu]�  	�          @�Q�@��
�Tz�k���\)C���@��
�J�H���\�{C�Q�                                    BxZul`  �          @�{@���{���{�6=qC�5�@���_\)������C��=                                    BxZu{  �          @�z�@���"�\>��?���C���@���   �����HC��\                                    BxZu��  
�          @�(�@����>k�?��HC�޸@���zᾮ{�8Q�C���                                    BxZu�R  �          @޸R@�=q��\)?���A8  C��@�=q��33?s33@��
C���                                    BxZu��  �          @���@��H���\����5p�C�AH@��H����!���z�C���                                    BxZu��  
(          @���@7����H�
=q���HC��@7�����X������C��                                    BxZu�D  �          @�33@��>aG�@Z�HA��@	��@����R@W�A�{C���                                    BxZu��  �          @�(�@�33@'�@|(�B�A�
=@�33?�=q@��RB$��A�=q                                    BxZu�  �          @�  @���@p�@mp�B�A��@���?�p�@�z�B
=A[�
                                    BxZu�6  �          @�(�@�=q��Q�@��B2{C�q�@�=q��G�@��\B*33C��{                                    BxZu��            @��@�  ���@���B��C���@�  ����@���B��C�G�                                    BxZv�  �          @��H@��׾�{@;�AϮC�g�@��׿�ff@0��A\C�                                    BxZv(  �          @��H@�=q�u@<��A�33C�� @�=q�u@333A��C��                                     BxZv*�  �          @ٙ�@�G�>Ǯ@W
=A��@u�@�G���@VffA��C��)                                    BxZv9t  �          @ڏ\@�{>�z�@O\)A�R@5@�{��@Mp�A�z�C��                                     BxZvH  �          @أ�@�ff>u@�  B�@)��@�ff�Tz�@�B�C�}q                                    BxZvV�  T          @أ�@�(���\@e�BffC�h�@�(����@UA�p�C�3                                    BxZvef  �          @��H@��H�z�@1�A�
=C�9�@��H�,��@
�HA���C��{                                    BxZvt  �          @��
@��H?��\?�(�A���A%G�@��H?��?�A�33@��                                    BxZv��  �          @޸R@��H@��@��A��
A��H@��H?�33@(��A���Arff                                    BxZv�X  �          @�
=@�=q?E�@�A�p�@�@�=q>aG�@��A�=q?�                                    BxZv��  �          @���@�ff��@QG�A���C�� @�ff�(�@/\)AŅC��                                    BxZv��  �          @�
=@���?�?�p�A#�@�  @���>L��?���A0z�?�p�                                    BxZv�J  T          @޸R@��
>#�
?\(�@�z�?���@��
�u?^�R@�  C��q                                    BxZv��  �          @��@�G�<��
>��H@�Q�>.{@�G���G�>�@z�HC���                                    BxZvږ  �          @߮@�{�E�>B�\?�=qC���@�{�J=q�#�
���
C��q                                    BxZv�<  �          @���@�
=�Y��=#�
>�Q�C���@�
=�Tz�.{��C���                                    BxZv��  �          @�G�@߮��R�   ��33C�y�@߮��׿&ff��Q�C��                                    BxZw�  �          @߮@�{��\�5���\C��@�{���R�Q���
=C��{                                    BxZw.  �          @�
=@��ͿB�\�(����C�� @��Ϳ녿G���C���                                    BxZw#�  T          @���@�\)���H��p��B�\C��)@�\)�\���H�~�RC�q�                                    BxZw2z  �          @ᙚ@���>aG�=��
?.{?���@���>B�\>\)?���?�=q                                    BxZwA   �          @�G�@���=�Q����=q?:�H@���=�Q�#�
��\)?E�                                    BxZwO�  �          @��@�\)?
=��Q��<(�@���@�\)?(�þL�Ϳ���@���                                    BxZw^l  �          @߮@޸R<#�
�������=��
@޸R>�������?�G�                                    BxZwm  �          @�{@�=q��  �0����{C��R@�=q�G��k����C���                                    BxZw{�  �          @߮@�{���ͿG���z�C�AH@�{�˅���R�#
=C�U�                                    BxZw�^  �          @�Q�@�G���ÿ5���C��{@�G�������,��C��                                    BxZw�  �          @�
=@�p��E�\)���C��{@�p��5������4z�C���                                    BxZw��  �          @�Q�@Ϯ�"�\�=p���G�C�T{@Ϯ�  �����5C�n                                    BxZw�P  �          @�\)@�G���׿:�H���C�s3@�G����R����-�C���                                    BxZw��  �          @�\)@�
=��33���R�EG�C�  @�
=�8Q�޸R�g\)C���                                    BxZwӜ  �          @�ff@У��zῌ���=qC�0�@У׿ٙ���{�W
=C���                                    BxZw�B  �          @�(�@���:�H�c�
��RC�Q�@���%�����\(�C��                                    BxZw��  �          @�z�@�G��QG���p��'33C�G�@�G��4z��33��\)C�H                                    BxZw��  �          @��
@���tz�333���
C���@���_\)��Q��e��C�                                    BxZx4  �          @��
@�33�{���Q��C�
C�y�@�33�l(���33�<��C�Q�                                    BxZx�  �          @�  @�G���    <��
C���@�G����ÿ�\)���C�Y�                                    BxZx+�  �          @أ�@���n�R��G��k�C�4{@���c�
��{�
=C���                                    BxZx:&  �          @�=q@�\)�h�þ\�L��C��f@�\)�Y������6�\C��H                                    BxZxH�  �          @�\)@��R�|(�@Q�A��C�o\@��R��@ffA�p�C���                                    BxZxWr  �          @�
=@�G���
=@L��A噚C���@�G���?�Q�A�p�C���                                    BxZxf              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZxt�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZx�d             @Ӆ@��
�0��@�A��
C��@��
�O\)?��A4Q�C��                                   BxZx�
  
�          @�{@�33�hQ�=L��>�
=C��@�33�`  �p���{C�c�                                   BxZx��  �          @�\)@����*�H�>�R��=qC���@��Ϳ�G��e�(�C���                                   BxZx�V  �          @��@���\)�'
=��G�C�H�@����
=�L������C��R                                    BxZx��  �          @��@�z��5�� ������C��\@�z����L(���{C�
                                    BxZx̢  �          @ۅ@�\)�$z��E���z�C�W
@�\)�У��j�H�
=C���                                    BxZx�H  
�          @��@��\���H��33�<z�C���@��\>u����C(�@>�R                                    BxZx��  T          @�\)@��R�'��XQ���G�C�/\@��R�˅�~{��C���                                    BxZx��  "          @�=q@�
=��\���
�S�\C��\@�
=?����G��Op�Aa�                                    BxZy:  �          @��H@�G�>�
=���
�R
=@���@�G�?�����G��A{A�                                    BxZy�  �          @�\)@�\)>������Xz�@`��@�\)?�\)����H�A�
=                                    BxZy$�  
�          @�@��׿
=����\{C��3@���?�ff��  �X�AiG�                                    BxZy3,  
�          @�=q@L�Ϳ�=q��
=�i33C�
@L�ͽ�����  �{z�C�R                                    BxZyA�  
�          @�\@Dz��
=��p��c(�C�4{@Dz��R���H�~�HC�H�                                    BxZyPx  �          @�\@J�H�(����\�^�C�9�@J�H�5�����zz�C��f                                    BxZy_  �          @ᙚ@L(������z��cffC��@L(������  �{  C�U�                                    BxZym�  �          @�=q@<(��(���Q��j��C���@<(������(�(�C��q                                    BxZy|j  T          @��@=p����������r{C�.@=p����
���u�C���                                    BxZy�  �          @�G�@HQ��3�
��33�T\)C��@HQ쿏\)����v��C��                                    BxZy��  
�          @�33@��׿�z����>Q�C�T{@���=�G����
�G��?�=q                                    BxZy�\  T          @�=q@x�ÿ��H�����OQ�C���@x�þ��
���
�az�C��                                    BxZy�  "          @�=q@|���33���R�K=qC�J=@|�;�
=����^��C���                                    BxZyŨ  
Z          @���@|���z����Jp�C�'�@|�;�ff�����^=qC��                                     BxZy�N  
�          @���@����   ��p��2�C��@�����  ��p��Kp�C�k�                                    BxZy��  T          @�@����N{�hQ���{C�]q@����33��z��Q�C���                                    BxZy�  
Z          @�@��R���������)�C�
@��R?z�H���R�%��A2�\                                    BxZz @  �          @߮@�G�?�z���ff�)z�A�=q@�G�@J�H�}p��
\)B(�                                    BxZz�  �          @��
@��\�aG��n�R���C���@��\>k��u��p�@*�H                                    BxZz�  �          @�@�(��@  �aG���33C���@�(���\)��\)��C��                                    BxZz,2  
�          @�
=@�G���R�Z=q��\C��\@�G���33�~�R�\)C��                                    BxZz:�  
�          @�{@����2�\�C33��p�C��@�����ff�n�R�Q�C�f                                    BxZzI~  
Z          @��@�ff�������  C�"�@�ff�y���
=��Q�C��H                                    BxZzX$  �          @�G�@������þ�z��{C���@�����Q��=q�X��C�ٚ                                    BxZzf�  �          @�  @��
���
�#�
���
C�<)@��
���Ϳ�G��O�
C��                                    BxZzup  "          @�  @�=q����B�\��{C��q@�=q���\��p��n{C�^�                                    BxZz�  
Z          @�G�@����  �k����C�/\@����Q������Q�C���                                    BxZz��  T          @׮@_\)�����j=q�
=C�
=@_\)�:=q���R�7��C�R                                    BxZz�b  
�          @���@���{��W���C��@���2�\����!p�C���                                    BxZz�  
�          @�G�@�
=�tz��<(�����C�~�@�
=�333�{��Q�C��f                                    BxZz��  "          @�Q�@�{�����0������C���@�{�C33�tz��	=qC�|)                                    BxZz�T  
�          @ۅ@�=q�`���h����C�<)@�=q�33��Q��%�C���                                    BxZz��  
�          @�Q�@��Q�@�A�\)C�{@��7�?�  A,(�C�R                                    BxZz�  �          @��@\�<��?���A4(�C��@\�Mp�>�
=@a�C�
                                    BxZz�F  T          @�{@�33�QG�=���?W
=C��f@�33�J=q�\(�����C�Q�                                    BxZ{�  �          @޸R@��H�Dzῌ����HC���@��H�'
=���H����C�`                                     BxZ{�  
�          @��
@����{��
���
C��
@�������0  ��  C��                                    BxZ{%8  
�          @ۅ@���*=q�����\C��3@�녿�33�7
=���HC���                                    BxZ{3�  
�          @��H@�  ��(��s33��C���@�  ��\)����C��                                    BxZ{B�  T          @أ�@�p��qG��p���ffC��@�p��<���N{�㙚C�#�                                    BxZ{Q*  T          @ٙ�@��
��녿���T��C�<)@��
�j�H�/\)���C���                                    BxZ{_�  �          @ٙ�@���h�ÿ�  �r=qC���@���=p��0  ����C��3                                    BxZ{nv  �          @ٙ�@���P��?��@�{C���@���QG���G��l��C��                                     BxZ{}            @�33@�p��S�
�s33��{C�g�@�p��8Q�����C�
=                                    BxZ{��  
�          @�33@����.�R������C�h�@�����(��7
=��{C���                                    BxZ{�h  
�          @�(�@�����
�\����C�B�@���+��tz��z�C���                                    BxZ{�  �          @ָR@��(���2�\��{C��\@���
=�\���  C���                                    BxZ{��  �          @љ�@}p��k��c�
�ffC��\@}p��������/��C�%                                    BxZ{�Z  
�          @�=q@���XQ�@�33BN��C��f@����  @w
=Bp�C�k�                                    BxZ{�   
�          @љ�@`������@&ffA���C��@`�����?��A  C��)                                    BxZ{�  "          @��
@\����
=@(��A��HC�^�@\������?��
A�C���                                    BxZ{�L  
�          @�Q�@u���H?�A�{C�7
@u��{>\@U�C�=q                                    BxZ| �  
�          @�=q@��H�`  ���
��G�C�Y�@��H�333�0  ����C�O\                                    BxZ|�  
�          @�G�@����33��  �  C�n@��ͿW
=��\)�4Q�C�'�                                    BxZ|>  �          @�G�@�33��\��p��&p�C��)@�33����=q�8ffC�޸                                    BxZ|,�  �          @�=q@����   ��=q�+��C���@��׾���ff�=G�C�Ff                                    BxZ|;�  T          @�33@����G������@33C��)@��>#�
��\)�J�\@�                                    BxZ|J0  T          @��
@��������  �Wz�C�z�@���?333����Z��A��                                    BxZ|X�  !          @��@\)�����=q�Z\)C�Y�@\)?:�H��(��]�\A$��                                    BxZ|g|  T          @�=q@N{?.{���R�t�\A>�R@N{@p���\)�V�B�\                                    BxZ|v"  "          @�p�@[��
=�\�p�RC�"�@[�?����
=�iA�ff                                    BxZ|��  �          @�@r�\��ff�����\�
C��)@r�\>��
���H�h
=@���                                    BxZ|�n  �          @�G�@aG�������z��d�C��\@aG�>�
=����n�\@�\)                                    BxZ|�  T          @�=q@S33��33�����h��C�� @S33>�(���ff�s  @�\)                                    BxZ|��  "          @أ�@fff��{��=q�]p�C��@fff>�����R�f��@�{                                    BxZ|�`  
�          @�{@z=q�Ǯ��Q��RQ�C�)@z=q>L����
=�^ff@8Q�                                    BxZ|�  T          @��@R�\��Q���  �a�\C���@R�\��Q�����v�C�8R                                    BxZ|ܬ  T          @�{@1녿��������r��C�w
@1�    ���H�RC��{                                    BxZ|�R  
�          @��@AG���
=�����mffC��R@AG�<��
��33�>���                                    BxZ|��  
�          @�\)@*�H�=q�Ǯ�p{C��@*�H��p����C�
=                                    BxZ}�  
(          @���@���(��У��~{C��@�þ��R��ff\)C�޸                                    BxZ}D  "          @�G�?�
=��H�ҏ\�\C�L�?�
=��\)��Q��
C��\                                    BxZ}%�  �          @�=q?���%�ҏ\\C�
=?�녾�(��ᙚQ�C��f                                    BxZ}4�  
�          @�R?����
�����C���?�>�Q����
� A2�\                                    BxZ}C6  
�          @�=q?�ff�p���ff{C�)?�ff��=q��(��C�)                                    BxZ}Q�  
�          @�=q?�ff�����H\C���?�ff>B�\��z�Q�@��                                    BxZ}`�  �          @��
?������ff�C�� ?��������338RC�˅                                    BxZ}o(  �          @�=q@���333��z��tz�C���@�ÿ.{��{�C�=q                                    BxZ}}�  
�          @���?������ٙ���C��
?���=#�
��z�L�?�(�                                   BxZ}�t              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ}�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ}��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ}�f              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ}�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ}ղ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ}�X              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ}��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~J              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~-�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~<<              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~J�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~Y�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~h.              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~v�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~�z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~�               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~�l              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~θ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~�^              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ	P              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ&�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ5B              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZC�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZR�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZa4              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZo�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ~�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�&              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�r              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZǾ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�d              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�
              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�V              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�.H              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�<�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�K�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�Z:              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�h�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�w�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��,              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��x              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��j              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�
              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�'N              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�5�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�D�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�S@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�a�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�p�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�2              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��$              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��p              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��b              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ� T              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�.�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�=�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�LF              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�Z�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�i�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�x8   2          @�\)?�{<��
��  B�>��?�{@=q�ڏ\BQ�                                    BxZ���            @�@.{?����H��A�33@.{@g
=��\)�S��BS�                                    BxZ���  	�          A ��@�H��Q�����{C��f@�H@(���33\B(�
                                    BxZ��*  
�          @���@��=���(��
@7�@��@{��{�{�B7�                                    BxZ���  �          @�33@=L����R?��@@(������~�B8��                                    BxZ��v  "          @��@2�\�:�H��{�C��@2�\?�{��G�Q�A�                                    BxZ��  "          @�@>�R�n{�ٙ��C�\)@>�R?�����
=��HA�=q                                    BxZ���  �          @�\)@Dz�У����
�y(�C��@Dz�?&ff��G�W
A@Q�                                    BxZ��h  T          @���@_\)��33�����h��C��
@_\)>�����p��y{@�33                                    BxZ��  �          @�R@i���%���R�Tz�C�H�@i������Ϯ�q�
C�P�                                    BxZ�
�  
�          @�  @��
�@����
=�;�C��=@��
���
����]��C��)                                    BxZ�Z  T          @���@����Z=q���\�+G�C���@������
���Q��C�4{                                    BxZ�(   "          @�@j=q�]p������?(�C�O\@j=q����(��i��C�aH                                    BxZ�6�  �          @�@��R��(���{�p�C��H@��R�/\)����7C�5�                                    BxZ�EL  "          @��H@�Q���ff��Q��  C��\@�Q��{�����C�C��                                    BxZ�S�  �          @�\@����z�H?�G�A��C�N@������þ�33�K�C��                                    BxZ�b�  "          @�ff@��H�[�@��A��C�˅@��H�}p�?n{@���C���                                    BxZ�q>  
�          @�p�@�=q�N�R>W
=?��HC�y�@�=q�G
=�^�R��G�C��                                    BxZ��  T          @�  @b�\�s�
@�B�C�t{@b�\��\)@+�A�G�C��{                                    BxZ���  �          @޸R@��fff@��
BP�C�AH@���@w�B
�RC�"�                                    BxZ��0  "          @�=q@q����
@r�\B{C�p�@q���(�@Q�A�G�C���                                    BxZ���  T          @�=q@�Q���
=?��HA?�
C��@�Q���p���=q���C�T{                                    BxZ��|  T          @�ff@��H���<��
>L��C�'�@��H���H�˅�Q�C��
                                    BxZ��"  �          @�
=@O\)�QG�����%��C�Y�@O\)��p���p��U
=C��f                                    BxZ���  �          @ۅ@3�
�Fff���\�OffC��@3�
��33����}ffC��R                                    BxZ��n  
�          @�=q@P  �=p���{�G�C��
@P  �����(��o\)C���                                    BxZ��  �          @�(�@q��E���(��4
=C�g�@q녿��
���
�Z�\C���                                    BxZ��  "          @ᙚ@a��7
=�����Ez�C���@a녿n{��{�i��C��R                                    BxZ�`  "          @���@����Y������� ��C�T{@��Ϳٙ������I�C��
                                    BxZ�!  
Z          @���@q�������\���C�|)@q�������J33C���                                    BxZ�/�  
�          @�z�?��\����u���C�o\?��\�~{��
=�R��C�޸                                    BxZ�>R  T          @�
=@���
=����ffC���@��Q������]Q�C�t{                                    BxZ�L�  �          @޸R@�
=��
��(��3�C��q@�
=���
��G��G  C��                                    BxZ�[�  
�          @�p�@������=q�3��C��3@�녾������J(�C��R                                    BxZ�jD  �          @�
=@�z��0  ��{�"G�C��)@�z῏\)��33�Ap�C��q                                    BxZ�x�  
�          @�(�@�p��%����.�C��@�p��Tz���ff�I��C��3                                    BxZ���  �          @߮@s�
�P����(��0�
C���@s�
������{�YffC��\                                    BxZ��6  "          @���@\)�Z�H���� {C���@\)��G���G��J(�C��                                    BxZ���  �          @�(�@�p��\)��\)�.��C��{@�p��
=q��ff�E�C��                                    BxZ���  �          @�(�@^�R������
�[(�C��H@^�R=��
���n33?�\)                                    BxZ��(  �          @�p�@b�\��=q��33�Z=qC�Q�@b�\>���z��k�@
=q                                    BxZ���  
�          @�=q@�z��%�k���C��@�z῜(����\�"�C��                                    BxZ��t  T          @��
@�z��(��}p���C��=@�z�޸R��z��`��C�c�                                    BxZ��  "          @�ff@�����\� ����ffC��@��ÿ���C33�ӮC��                                     BxZ���  T          @أ�@���p��C�
���
C��3@����ff�mp���C�W
                                    BxZ�f  �          @�p�@���~{�����0  C��@���i�������o\)C�9�                                    BxZ�  
�          @���@��\��  ?uA z�C�>�@��\��녿
=��{C��                                    BxZ�(�  �          @���@�����R?Y��@�(�C�!H@����ff�n{��\)C�*=                                    BxZ�7X  T          @�
=@�������>�(�@i��C�|)@�����p���G����C�Ф                                    BxZ�E�  �          @ٙ�@��
�w
=?uAz�C�&f@��
�}p���p��Mp�C���                                    BxZ�T�  "          @�Q�@���p  @�\A�33C���@������?h��@��C��f                                    BxZ�cJ  "          @�(�@��H��33@�A�C���@��H���?Y��@ۅC��                                    BxZ�q�  
�          @��
@�Q��y��@��A�z�C�]q@�Q����?G�@ʏ\C���                                    BxZ���  �          @��@�����{@33A��RC�R@�����(�?
=q@�  C���                                    BxZ��<  
�          @�
=@�p���
=?n{@�Q�C��H@�p�����E���C��                                    BxZ���  �          @��@�(���z�?�ffA*�\C�}q@�(���G��Ǯ�Mp�C��                                    BxZ���  T          @�{@�=q��  ?333@�=qC�f@�=q����@  ��C��                                    BxZ��.  �          @�p�@��R��\)�8Q�����C�xR@��R�p  �����
C�!H                                    BxZ���  T          @�@���G��B�\��=qC��@����R��\�n=qC��                                    BxZ��z  �          @�
=@�Q����׿aG����C�33@�Q����R�=q��  C��)                                    BxZ��   �          @��H@�(���\)���
�%�C���@�(���G��1G����C��f                                    BxZ���  T          @�@��H��ff��z��  C�l�@��H��G��/\)��(�C�|)                                    BxZ�l  
�          @��
@�  ���������p�C��=@�  ��\)��p��a��C��f                                    BxZ�  
�          @��
@��\��\)<�>uC�Ǯ@��\���R��\)�R�RC��{                                    BxZ�!�  
�          @�33@������>�?�=qC���@������R��=q�-p�C�l�                                    BxZ�0^  �          @��H@������?�\@��C��{@������\�z�H��ffC���                                    BxZ�?  �          @�p�@�p���
=?k�@�C���@�p����׿(����C�j=                                    BxZ�M�  
�          @�(�@�ff�i��@   A��C��@�ff��33?.{@�{C���                                    BxZ�\P  "          @�\)@�Q����?L��@��
C�y�@�Q���{��R���C�`                                     BxZ�j�  
(          @�G�@����
==��
?!G�C�7
@����  ��z��8��C��                                    BxZ�y�  T          @ָR@�=q��zᾸQ��EC���@�=q���׿���w�C��f                                    BxZ��B  
�          @�@�{��\)�s33���C���@�{�k���\��ffC��3                                    BxZ���  �          @ʏ\@j=q����5��ԏ\C�'�@j=q�H����33�!ffC���                                    BxZ���  S          @���@�=q�W�>W
=?���C�c�@�=q�P  �k��C�ٚ                                    BxZ��4  �          @�\)@���L(�?��A9��C�  @���^{>��@�C���                                    BxZ���  �          @�ff@����p  ?��@�
=C��@����o\)�+���G�C��R                                    